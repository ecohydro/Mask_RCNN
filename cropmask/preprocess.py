import os
import copy
import skimage.io as skio
import warnings
import geopandas as gpd
from rasterio import features
import rasterio
import us
import xarray
import rioxarray
from numpy import uint8
from cropmask.label_prep import rio_bbox_to_polygon
from cropmask.misc import parse_yaml, make_dirs, img_to_png, label_to_png
from cropmask import sequential_grid, label_prep
from cropmask import io_utils 
from cropmask import coco_convert

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
    SCENE = os.path.join(DATASET, params['dirs']["scene"])
    TRAIN = os.path.join(DATASET, params['dirs']["train"])
    NEG_BUFFERED = os.path.join(DATASET, params['dirs']["neg_buffered_labels"])

    directory_list = [
            DATASET,
            SCENE,
            TRAIN,
            NEG_BUFFERED,
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
        self.SCENE = os.path.join(self.DATASET, params['dirs']["scene"])
        self.TRAIN = os.path.join(self.DATASET, params['dirs']["train"])
        self.TILES = os.path.join(self.DATASET, params['dirs']["tiles"])
        
        # scene specific paths and variables
        self.scene_basename = os.path.splitext(os.path.basename(self.scene_dir_path))[0]
        self.label_basename = os.path.splitext(os.path.basename(self.source_label_path))[0]
        tifname = self.label_basename + "_" + self.scene_basename + ".tif"
        self.rasterized_label_path = os.path.join(self.NEG_BUFFERED, tifname)
        self.band_list = [] # the band indices
        self.meta = {} # meta data for the scene
        self.chip_img_paths = [] # list of chip ids of form [scene_id]_[random number]
        self.chip_label_paths = [] # list of chip labels of form [scene_id]_[random number]
        self.small_area_filter = params['label_vals']['small_area_filter']
        self.neg_buffer = params['label_vals']['neg_buffer']
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
        scene_arr.rio.to_raster(scene_path)
        return self
            
    def negative_buffer_and_small_filter(self):
        """For preprcoessing reference dataset"""
        shp_frame = gpd.read_file(self.source_label_path)
        shp_frame = shp_frame.loc[shp_frame.geometry.area > self.small_area_filter]
        shp_frame = shp_frame.buffer(self.neg_buffer)
        return shp_frame.loc[shp_frame.is_empty==False]
            
    def tile_scene_and_vector(self, neg_buffer, small_area_filter):
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
        shp_frame = self.negative_buffer_and_small_area_filter()
        scene_vector_intersection_bounds = shp_frame.intersection(rio_bbox_to_polygon(self.bounds)).bounds
        raster_tiler = sol.tile.raster_tile.RasterTiler(dest_dir=os.path.join(self.TILES,"image_tiles"),  # the directory to save images to
                                                src_tile_size=(512, 512),  # the size of the output chips
                                                verbose=True,
                                                nodata=-9999,
                                                resampling="bilinear",
                                                tile_bounds=scene_vector_intersection_bounds)
        
        self.geojson_tile_dir = os.path.join(self.TILES,"geojson_tiles")
        solaris.tile.vector_tile(dest_dir=self.geojson_tile_dir,  # the directory to save images to
                                                src_tile_size=(512, 512),  # the size of the output chips
                                                verbose=True,
                                                tile_bounds=scene_vector_intersection_bounds)
        
        # everything below needs to be in own func
        
        for file_path in os.listdir(self.geojson_tile_dir):
            gdf = gpd.read_file(file_path)
            
            sol.vector.mask.instance_mask(gdf, out_file=None, reference_im=None, geom_col='geometry', do_transform=None, affine_obj=None, shape=(512, 512), out_type='int', burn_value=1, burn_field=None) # https://github.com/CosmiQ/solaris/pull/262/files

    
    def imgs_to_pngs(self):
        """
        Extracts individual instances into their own tif files. Saves them
        in each folder ID in train folder. If an image has no instances,
        saves it with a empty mask.
        """
        for tif_path in self.chip_img_paths:
            # for imgs with no instances, creates empty mask
            # only runs connected comp if there is at least one instance
            jpeg_path = os.path.splitext(tif_path)[0] + ".png"
            img_to_png(tif_path, jpeg_path)
            
        return self
    
    def labels_to_pngs(self):
        """
        Extracts individual instances into their own tif files. Saves them
        in each folder ID in train folder. If an image has no instances,
        saves it with a empty mask.
        """
        for tif_path in self.chip_label_paths:
            # for imgs with no instances, creates empty mask
            # only runs connected comp if there is at least one instance
            jpeg_path = os.path.splitext(tif_path)[0] + ".png"
            label_to_png(tif_path, jpeg_path)
            
        return self

    def run_single_scene(self):
    
        band_list = self.yaml_to_band_index()
        
        product_list = self.get_product_paths(band_list)
        
        self.load_meta_and_bounds(product_list)
        
        self.stack_and_save_bands()
        
        self.negative_buffer_and_small_filter(-31, 100)
        
        self.grid_images()
        
        self.connected_components()
        
        # train test split is done outside of this function to accomodate multiple scenes
        