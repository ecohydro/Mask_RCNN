### currently random, useful functions
from PIL import Image
from skimage import img_as_ubyte, exposure
from skimage.io import imread
from rasterio.plot import reshape_as_image
import numpy as np
import yaml
import shutil
import os
import random

def percentile_rescale(arr):
    """
    Rescales and applies other exposure functions to improve image vis. 
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    """
    rescaled_arr = np.zeros_like(arr)
    for i in range(0, arr.shape[-1]):
        val_range = (np.percentile(arr[:, :, i], 1), np.percentile(arr[:, :, i], 99))
        rescaled_channel = exposure.rescale_intensity(arr[:, :, i], val_range)
        rescaled_arr[:, :, i] = rescaled_channel
        # rescaled_arr= exposure.adjust_gamma(rescaled_arr, gamma=1) #adjust from 1 either way
    #     rescaled_arr= exposure.adjust_sigmoid(rescaled_arr, cutoff=.50) #adjust from .5 either way
    return rescaled_arr

def remove_dirs(directory_list):
    """
    Removes all files and sub-folders in each folder.
    """

    for f in directory_list:
        if os.path.exists(f):
            shutil.rmtree(f)

def max_normalize(arr):
    arr *= 255.0 / arr.max()
    return arr

def parse_yaml(input_file):
    """Parse yaml file of configuration parameters."""
    with open(input_file, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

def make_dirs(directory_list):

    # Make directory and subdirectories
    for d in directory_list:
        try: 
            os.mkdir(d)
        except:
            print("Whole directory list: ", directory_list)
            print("The directory "+d+" exists already. Check it and maybe delete it or change config.")
            raise FileExistsError

                    
def train_test_split(chip_dir_path, seed, split_proportion):
    """Takes a directory of gridded images and labels and returns the ids 
    of the train_validate set and the test set.
    Each sample folder contains an images and corresponding masks folder."""
    random.seed(seed)
    id_list = next(os.walk(chip_dir_path))[1]
    k = round(split_proportion * len(id_list))
    test_list = random.sample(id_list, k)
    train_validate_list = list(set(id_list) - set(test_list))
    return train_validate_list, test_list


def get_arr_channel_mean(chip_folder, channel):
    """
    Calculate the mean of a given channel across all training samples.
    """

    means = []
    train_list = next(os.walk(chip_folder))[1]
    for i, fid in enumerate(train_list):
        im_folder = os.path.join(chip_folder, fid, "image")
        im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
        arr = skio.imread(im_path)
        arr = arr.astype(np.float32, copy=False)
        # added because no data values different for wv2 and landsat, need to exclude from mean
        nodata_value = 0 # best to do no data masking up front and set bad qa bands to 0 rather than assuming 0 is no data. This is assumed from looking at no data values at corners being equal to 0
        arr[arr == nodata_value] = np.nan
        means.append(np.nanmean(arr[:, :, channel]))
    return np.mean(means)


def img_to_jpeg(tif_path, jpeg_path):
    """
    Converts processed tif chip images into pngs
    """
    arr = imread(tif_path)
    
    arr = np.dstack([arr[:,:,2],arr[:,:,1],arr[:,:,0]])

    img = Image.fromarray(img_as_ubyte(exposure.equalize_adapthist(arr)), mode='RGB')
    
    img.save(jpeg_path, format='jpeg')
    
    
def label_to_jpeg(tif_path, jpeg_path):
    """
    Converts processed tif chip labels into pngs
    """
    arr = imread(tif_path)
    
    if len(arr.shape) > 2:
        arr = np.any(arr, axis=2)

    img = Image.fromarray(img_as_ubyte(arr))
    
    img.save(jpeg_path, format='jpeg')