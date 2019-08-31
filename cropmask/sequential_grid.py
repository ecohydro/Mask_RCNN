#sequential gridding of a single landsat scene is currently faster than threading or multiprocessing in grid.py
import os
import pathlib
from itertools import product
import rasterio
from rasterio import windows
from multiprocessing.dummy import Pool as ThreadPool
from geopandas import GeoSeries
from cropmask.label_prep import rio_bbox_to_polygon
from rasterio import coords

def get_tiles_for_threaded_map(ds, label_boundary_gdf, width, height, usable_threshold):
    """
        Returns a list of tuple where each tuple is the window and transform information for the image chip.
                
        Args:
            ds (rasterio dataset): A rasterio object read with open()
            label_boundary_gdf: a geodataframe with a single polygon geometry demarcating where labels have been exhaustively collected
            width (int): the width of a tile/window/chip
            height (int): height of the tile/window/chip
        Returns:
            a list of tuples, where the first element of the tuple is a window and the next is the transform
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    chip_list = []
    def get_win(ds, label_boundary_gdf, col_off, row_off, width, height, big_window, usable_threshold):
        """Helper func to get the window and transform for a particular section of an image
        Args:
            ds (rasterio dataset): A rasterio object read with rasterio.open()
            label_boundary_gdf: a geodataframe with a single polygon geometry demarcating where labels have been exhaustively collected
            col_off (int): the column of the window, the upper left corner
            row_off (int): the row of the window, the upper left corner
            width (int): the width of a tile/window/chip
            height (int): height of the tile/window/chip
            big_window (rasterio.windows.Window): used to deal with windows that extend beyond the source image
        Returns:
            Returns the bounds of each image chip/tile as a rasterio window object as well as the transform
            as a tuple like (rasterio.windows.Window, transform)
        """
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        #transform = windows.transform(window, ds.transform)
        window_bbox = coords.BoundingBox(*windows.bounds(window, ds.transform))
        window_poly = rio_bbox_to_polygon(window_bbox)
        arr = ds.read(window=window)
        arr[arr < 0] = 0
        pixel_count = arr.shape[0] * arr.shape[1]
        nodata_pixel_count = (arr == 0).sum()
        if label_boundary_gdf.contains(GeoSeries(window_poly)).values[0] and nodata_pixel_count / pixel_count < usable_threshold:
            return(window, ds.transform) 
        else:    
            pass
    chip_list = list(map(lambda x: get_win(ds, label_boundary_gdf, x[0], x[1], width, height, big_window, usable_threshold), offsets))
    return [x for x in chip_list if x is not None]

def write_by_window(ds, out_dir, fid, chip_id, output_name_template, meta, window, transform):
    """Writes out a window of a larger image given a window and transform. 
    Args:unioned
        ds (rasterio dataset): A rasterio object read with open()
        out_dir (str): the output directory for the image chip
        output_name_template (str): string with curly braces for naming tiles by indices for uniquiness
        meta (dict): meta data of the ds 
        window (rasterio.windows.Window): the window to read and write
        transform (rasterio transform object): the affine transformation for the window
    Returns:
        Returns the outpath of the window that has been written as a tile
    """
    meta['transform'] = transform
    meta['width'], meta['height'] = window.width, window.height
    outfolder = os.path.join(out_dir, chip_id.format(int(window.col_off), int(window.row_off)), fid)
    if os.path.isdir(outfolder) is False:
        path = pathlib.Path(outfolder)
        path.mkdir(parents=True, exist_ok=False)
    outpath = os.path.join(outfolder, output_name_template.format(int(window.col_off), int(window.row_off)))
    with rasterio.open(outpath, 'w', **meta) as outds:
            outds.write(ds.read(window=window))
    return outpath

def grid_images_rasterio_sequential(in_path, out_dir, fid, chip_id, label_boundary_gdf, output_name_template, chip_list, grid_size=512):
    """Combines get_tiles_for_threaded_map, map_threads, and write_by_window to write out tiles of an image
    Args:
        in_path (str): Path to a raster for which to read with raterio.open()
        out_dir (str): the output directory for the image chip
        label_boundary_gdf: a geodataframe with a single polygon geometry demarcating where labels have been exhaustively collected
        output_name_template (str): string with curly braces for naming tiles by indices for uniquiness
        grid_size (int): length in pixels of a side of a single window/tile/chip
    Returns:
        Returns the outpaths of the tiles.
    """
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        out_paths = list(map(lambda x: write_by_window(src, out_dir, fid, chip_id, output_name_template, meta, x[0], x[1]), chip_list)) #change to map_threads for threading but currently fails partway
    return out_paths
