import os
import copy
import skimage.io as skio
from skimage import img_as_ubyte, exposure
import warnings
import geopandas as gpd
from rasterio import features
import rasterio
from shapely.ops import unary_union
from shapely.geometry import shape, box
import us
import xarray
import rioxarray
from numpy import uint8
from cropmask.label_prep import rio_bbox_to_polygon
from cropmask.misc import parse_yaml, make_dirs, img_to_png, label_to_png
from cropmask import label_prep
from cropmask import io_utils 
from cropmask import coco_convert
from PIL import Image as pilimg
import solaris as sol
from tqdm import tqdm
import numpy as np
from pathlib import Path

def setup_dirs(param_path):
    """
    This folder structure is used for each unique pre processing and modeling
    workflow and is made unique by specifying a unique DATASET name
    or ROOT path (if working on a different container.). 

    ROOT should be the path to the azure container mounted with blobfuse, 
    and should already exist. The RESULTS folder should be created in a folder named from param["results"], and this should also already exist.
    """
    
    params = parse_yaml(param_path)
    
    # the folder structure for the unique run
    ROOT = params['dirs']["root"]
    DATASET = os.path.join(ROOT, params['dirs']["dataset"])
    SCENE = os.path.join(DATASET, "scene")
    TILES = os.path.join(DATASET, "tiles")
    COCO = os.path.join(DATASET, "coco")
    image_tile_dir = os.path.join(TILES,"image_tiles")
    geojson_tile_dir = os.path.join(TILES,"geojson_tiles")
    label_tile_dir = os.path.join(TILES, "label_tiles")
    jpeg_tile_dir = os.path.join(TILES, "jpeg_tiles")

    directory_list = [
            DATASET,
            SCENE,
            COCO,
            TILES,
            image_tile_dir,
            geojson_tile_dir,
            label_tile_dir,
            jpeg_tile_dir
        ]
    make_dirs(directory_list)
    return directory_list

class PreprocessWorkflow():
    """
    Worflow for loading and gridding a single satellite image and reference dataset of the same extent.
    """
    
    def __init__(self, param_path, scene_dir_path='nopath/', source_label_path='nopath/'):
        params = parse_yaml(param_path)
        self.params = params
        self.source_label_path = source_label_path
        self.scene_dir_path = scene_dir_path # path to the unpacked tar archive on azure storage
        self.scene_id = self.scene_dir_path.split("/")[-2] # gets the name of the folder the bands are in, the scene_id
        
         # the folder structure for the unique run
        self.ROOT = params['dirs']["root"]
        self.DATASET = os.path.join(self.ROOT, params['dirs']["dataset"])
        self.SCENE = os.path.join(self.DATASET, "scene")
        self.COCO = os.path.join(self.DATASET, "coco")
        self.TILES = os.path.join(self.DATASET, "tiles")
        self.image_tile_dir = os.path.join(self.TILES,"image_tiles")
        self.geojson_tile_dir = os.path.join(self.TILES,"geojson_tiles")
        self.label_tile_dir = os.path.join(self.TILES, "label_tiles")
        self.jpeg_tile_dir = os.path.join(self.TILES, "jpeg_tiles")
        # scene specific paths and variables
        self.band_list = [] # the band indices
        self.meta = {} # meta data for the scene
        self.small_area_filter = params['label_vals']['small_area_filter']
        self.ag_class_int = params['label_vals']['ag_class_int'] # TO DO, not implemented but needs to be for multi class
        self.dataset_name = params['image_vals']['dataset_name']
        self.grid_size = params['image_vals']['grid_size']
        self.usable_threshold = params['image_vals']['usable_thresh']
        self.split = params['image_vals']['split']

    def yaml_to_band_index(self):
        """Parses config booleans to a list of band indexes to be stacked.
        For example, Landsat 5 has 6 bands (7 if you count band 6, thermal) 
        that we can use for masking.
        Args:
            params (dict): The configuration dictionary that is read with yaml.
        Returns:
            list: A list of strings for the band numbers, starting from 1. For Landsat 5 1 would
            represent the blue band, 2 green, and so on. For Landsat 8, band 1 would be coastal blue,
            band 2 would be blue, and so on.
            
            See https://landsat.usgs.gov/what-are-band-designations-landsat-satellites
        .. _PEP 484:
            https://www.python.org/dev/peps/pep-0484/
        """
        if self.params["image_vals"]["dataset_name"] == "landsat-5":
            bands = self.params["landsat_bands_to_include"]
        for i, band in enumerate(bands):
            if list(band.values())[0] == True:
                self.band_list.append(str(i+1))
        return self.band_list
    
    def get_product_paths(self, band_list):
        # Load image
        product_list = os.listdir(self.scene_dir_path)
        # below works because only products that are bands have a int in the 5th to last position
        filtered_product_list = [band for band in product_list if band[-5] in band_list and ('band' in band or "SRB" in band)]
        filtered_product_list = sorted(filtered_product_list)
        filtered_product_paths = [os.path.join(self.scene_dir_path, fname) for fname in filtered_product_list]
        return filtered_product_paths
    
    def load_meta_and_bounds(self, product_paths):
        # get metadata and edit meta obj for stacked raster
        with rasterio.open(product_paths[0]) as rast:
                meta = rast.meta.copy()
                meta.update(compress="lzw")
                meta["count"] = len(product_paths)
                self.meta=meta
                self.bounds = rast.bounds
        return self
    
    def stack_and_save_bands(self):
        """Load the landsat bands specified by yaml_to_band_index and returns 
        a [H,W,N] Numpy array for a single scene, where N is the number of bands 
        and H and W are the height and width of the original band arrays. 
        Channels are ordered in band order.

        Args:
            scene_dir_path (str): The path to the scene directory. The dir name should be the standard scene id that is the same as
            as the blob name of the folder that has the landsat product bands downloaded using lsru or
            download_utils.
            band_list (str): a list of band indices to include

        Returns:
            ndarray:k 

        .. _PEP 484:k 
            https://www.python.org/dev/peps/pep-0484/

        """
        
        product_paths = self.get_product_paths(self.band_list)
        scene_arr = io_utils.read_bands_lsr(product_paths)
        scene_name = os.path.basename(product_paths[0])[:-10] + ".tif"
        scene_path = os.path.join(self.SCENE, scene_name)
        self.scene_path = scene_path
        scene_arr.rio.to_raster(scene_path, dtype="int16")
        return self
            
    def filter_subset_vector_gdf(self, bounds_gdf):
        """For preprcoessing reference dataset"""
        shp_frame = gpd.read_file(self.source_label_path)
        shp_frame = shp_frame.to_crs(self.meta['crs'])
        shp_frame.crs = shp_frame.crs.to_wkt()# geopandas can't save dfs with crs in rasterio format
        shp_frame = shp_frame.cx[self.bounds.left:self.bounds.right,self.bounds.bottom:self.bounds.top] # reduces computation to only operate on labels intersecting image
        shp_frame = self.subtract_no_data_regions(shp_frame, bounds_gdf)
        shp_frame = shp_frame.loc[shp_frame.geometry.area > self.small_area_filter]
        shp_series = shp_frame.loc[shp_frame.is_empty==False]
        shp_frame.loc[shp_frame.is_valid==False, 'geometry'] = shp_frame[shp_frame.is_valid==False].buffer(0) # fix self intersections so that there is no topology error
        return shp_frame
    
    def subtract_no_data_regions(self, shp_frame, bounds_gdf):
        src = rasterio.open(self.scene_path)
        
        #removing labels by nodata boundaries in image
        polys = list(rasterio.features.shapes(src.read(), mask=(src.read() == src.nodata), connectivity=4, transform=src.transform))
        nodata_shapes = [shape(poly[0]) for poly in polys]
        boundary = gpd.GeoDataFrame(geometry=[unary_union(nodata_shapes)])
        src.close()
        shp_frame = gpd.overlay(shp_frame, boundary, how='difference')
        
        # removing labels by aoi boundary
        bounds_gdf_edge = bounds_gdf.copy()
        bounds_gdf_edge['geometry'] = bounds_gdf_edge.geometry.buffer(5000)
        bounds_gdf_edge = gpd.overlay(bounds_gdf_edge, bounds_gdf, how='difference')
        shp_frame = gpd.overlay(shp_frame, bounds_gdf_edge, how='difference')
        
        # removing labels by boundaries of image
        im_box_gdf = gpd.GeoDataFrame(geometry = [box(self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top)], crs = src.crs)
        im_box_edge = im_box_gdf.copy()
        im_box_edge['geometry'] = im_box_gdf.geometry.buffer(5000)
        im_box_edge = gpd.overlay(im_box_edge, im_box_gdf, how='difference')
        shp_frame = gpd.overlay(shp_frame, im_box_edge, how='difference')
        return shp_frame
    
    def get_vector_bounds_poly(self):
        #specify zipped shapefile url
        nebraska_url = us.states.NE.shapefile_urls('state')
        gdf = io_utils.zipped_shp_url_to_gdf (nebraska_url)
        return gdf.to_crs(self.meta['crs'])
            
    def tile_scene_and_vector(self):
        """
        Applies a negative buffer to labels since some are too close together and 
        produce conjoined instances when connected components is run (even after 
        erosion/dilation). This may not get rid of all conjoinments and should be adjusted.
        It relies too on the source projection of the label file to calculate distances for
        the negative buffer. It's assumed that the projection is in meters and that a negative buffer in meter units 
        will work with this projection.
        
        Args:
            source_label_path (str): the path to the reference shapefile dataset. Should be the same extent as a Landsat scene
            neg_buffer (float): The distance in meters to use for the negative buffer. Should at least be 1 pixel width.
            small_area_filter (float): The area thershold to remove spurious small fields. Particularly useful to remove fields                                         to small to be commercial agriculture

        Returns rasterized labels that are ready to be gridded
        """
        bounds_gdf= self.get_vector_bounds_poly()
        bounds_poly = bounds_gdf['geometry'].iloc[0]
        shp_frame = self.filter_subset_vector_gdf(bounds_gdf)
        
        self.raster_tiler = sol.tile.raster_tile.RasterTiler(dest_dir=self.image_tile_dir,  # the directory to save images to
                                                src_tile_size=(self.grid_size, self.grid_size),  # the size of the output chips
                                                verbose=True,
                                                aoi_boundary=bounds_poly,
                                                nodata= -9999.0)
        self.raster_bounds_crs = self.raster_tiler.tile(self.scene_path, restrict_to_aoi=True, nodata_threshold = .60)
        
        self.vector_tiler = sol.tile.vector_tile.VectorTiler(dest_dir=self.geojson_tile_dir,
                                                verbose=True) # check crs messes up non epsg crs if dest_Crs is set in instance creation of vector tiler
        self.vector_tiler.tile(shp_frame, tile_bounds=self.raster_tiler.tile_bounds, tile_bounds_crs = self.raster_bounds_crs, dest_fname_base=os.path.basename(self.scene_path).split(".tif")[0])
        self.geojson_tile_paths = self.vector_tiler.tile_paths
        self.raster_tile_paths = self.raster_tiler.tile_paths
        return self
        
    def for_each_img_tile(self, clamp_low, clamp_high):
        self.raster_tiler.fill_all_nodata(0) # filling needs to occur before rescaling and resaving as jpeg.
        self.all_chip_stats = {}
        for img_tile, geojson_tile in zip(tqdm(sorted(self.raster_tile_paths)), sorted(self.geojson_tile_paths)):
            self.geojson_to_mask(img_tile,geojson_tile)
            self.rescale_and_save(img_tile, clamp_low, clamp_high)
        return self.all_chip_stats # stats from the jpeg chips after filling nodata and rescaling

    def geojson_to_mask(self, img_tile, geojson_tile):
        fid = os.path.basename(geojson_tile).split(".geojson")[0]
        rasterized_label_path = os.path.join(self.label_tile_dir, fid + ".tif")
        try:
            gdf = gpd.read_file(geojson_tile)
        except:
            print(f"probably DriverError, check {geojson_tile} and {img_tile}")
        gdf.crs = self.raster_bounds_crs # add this because gdfs can't be saved with wkt crs
        arr = sol.vector.mask.instance_mask(gdf, out_file=rasterized_label_path, reference_im=img_tile, 
                                          geom_col='geometry', do_transform=None,
                                          out_type='int', burn_value=1, burn_field=None) # this saves the file, unless it is empty in which case we deal with it below.
        if not arr.any(): # in case no instances in a tile we save it with "empty" at the front of the basename
            with rasterio.open(img_tile) as reference_im:
                meta = reference_im.meta.copy()
                reference_im.close()
            meta.update(count=1)
            meta.update(dtype='uint8')
            if isinstance(meta['nodata'], float):
                meta.update(nodata=0)
            rasterized_label_path = os.path.join(self.label_tile_dir, "empty_" + fid + ".tif")
            with rasterio.open(rasterized_label_path, 'w', **meta) as dst:
                dst.write(np.expand_dims(arr, axis=0))
                dst.close()
                
    def rescale_and_save(self, img_tile, clamp_low, clamp_high):
        fid = os.path.basename(img_tile).split(".tif")[0]
        jpeg_path = os.path.join(self.jpeg_tile_dir, fid + ".jpg")
        img_array = skio.imread(img_tile)
        img_array = exposure.rescale_intensity(img_array, in_range=(clamp_low, clamp_high))  # Landsat 5 ARD .25 and 97.75 percentile range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_array = img_as_ubyte(img_array)
        img_pil = pilimg.fromarray(img_array)

        # Export chip images
        with open(Path(jpeg_path), 'w') as dst:
            img_pil.save(dst, format='JPEG', subsampling=0, quality=100)

        self.all_chip_stats[jpeg_path] = {'mean': img_array.mean(axis=(0, 1)),
                                         'std': img_array.std(axis=(0, 1))}

