import xarray as xr
import rioxarray
import glob
import os
import numpy as np
import requests
import geopandas as gpd
import fiona
from pathlib import Path
import tarfile

def open_rasterio_lsr(path):
    """Reads in a Landsat surface reflectance band and correctly assigns the band metadata.

    Args:
        path (str): Path of form 
            .../LT05_L1TP_032031_20050406_20160912_01_T1_sr_band5.tif.

    Returns:
        bool: Returns an xarray data array

    """
    
    band_int = int(os.path.splitext(path)[0][-1])
    data_array = xr.open_rasterio(path, chunks={'band': 1}) #chunks makes i lazyily executed
    band_val = data_array['band']
    band_val.data = np.array((band_int, ))
    return data_array

def read_bands_lsr(path_list):
    """
    Concatenates a list of landsat paths into a single data array.
    
    Args:
        path_list (str): Paths of form 
            .../LT05_L1TP_032031_20050406_20160912_01_T1_sr_band5.tif or 
            ARD format
            LT05_CU_012007_20050515_20190108_C01_V01_SRB5.tif
            in a list.

    Returns:
        bool: Returns an xarray data array
    """
    
    band_arrs = [open_rasterio_lsr(path) for path in path_list]
    return xr.concat(band_arrs, dim="band")

def write_xarray_lsr(xr_arr, fpath):
    xr_arr.rio.to_raster(fpath)

def read_scenes(scenes_folder_pattern):
    """
    Reads in multiple Landsat surface reflectance scenes given a regex pattern for ARD scene folders.

    Args:
        path (str): Path of form "../*".

    Returns:
        bool: Returns an xarray data array with dimensions for x, y, band, and time.
    """

    scene_folders = glob.glob(scenes_folder_pattern)
    # only select files that contain a band
    sr_paths = [glob.glob(scene_folder+'/*SRB*') for scene_folder in scene_folders]  
    xr_arrs = [read_bands_lsr(paths) for paths in sr_paths]
    return xr.concat(xr_arrs, dim="time")

def zipped_shp_url_to_gdf(url):
    """Opens a zipped shapefile in memory as a GeoDataFrame. Currently used for
    opening nebraska state shapefile from US Census as gdf."""
    
    request = requests.get(url)

    b = bytes(request.content)
    with fiona.BytesCollection(b) as f:
        crs = f.crs
        return gpd.GeoDataFrame.from_features(f, crs=crs)
    
def untar_all(in_folder, out_folder):
    folder = Path(in_folder)

    tar_gen = folder.glob("*tar")

    for f in tar_gen:

        otar = tarfile.open(f)

        otar.extractall(os.path.join(out_folder, f.stem))