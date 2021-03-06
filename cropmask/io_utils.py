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
    data_array = rioxarray.open_rasterio(path, chunks={'band': 1}) #chunks makes i lazyily executed
    data_array = data_array.sel(band=1).drop("band") # gets rid of old coordinate dimension since we need bands to have unique coord ids
    data_array["band"] = band_int # makes a new coordinate
    data_array = data_array.expand_dims({"band":1}) # makes this coordinate a dimension
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


def read_scenes(scene_folders):
    """
    Reads in multiple Landsat surface reflectance scenes given a regex pattern for ARD scene folders.

    Args:
        path (str): Path of form "../*".

    Returns:
        list: Returns a list of xarray data arrays
    """
    # only select files that contain a band
    sr_paths = [glob.glob(scene_folder+'/*SRB*') for scene_folder in scene_folders]  
    xr_arrs = [read_bands_lsr(paths) for paths in sr_paths]
    return xr_arrs

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