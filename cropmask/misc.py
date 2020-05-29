### currently random, useful functions
from PIL import Image
from skimage import img_as_ubyte, exposure
from skimage.io import imread, imsave
from rasterio.plot import reshape_as_image
import numpy as np
import pandas as pd
import yaml
import shutil
import os
import random
import warnings
from pathlib import Path
import json
from cropmask.coco_convert import save_coco_annotation

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

def read_and_mean(path):
    arr = imread(str(path))
    fill_values = np.mean(arr, axis=(0, 1))
    return fill_values
    
def read_and_variance(path):
    arr = imread(str(path))
    fill_values = np.var(arr, axis=(0, 1))
    return fill_values

def calc_stat(df, func, include_empty=False):
    """
    func should operate on a single array of order [band, H, W]
    """
    df = df.copy()
    new_df = df[df.is_empty == include_empty]
    new_df['fill_values'] = new_df['jpeg_tiles'].apply(func)
    new_df = pd.DataFrame(new_df.fill_values.tolist(), columns=['0', '1', '2'])
    return new_df

def max_norm_channels(arr):
    """
    arr must be of shape (w, h, channel)
    """
    arr = arr.copy()
    maxes = np.nanmax(arr, axis=(0,1))
    return arr / maxes

def make_vis_im(img):
    img = img[:, :, ::-1]
    normalized = max_norm_channels(np.where(img < 0, 0, img))
    rescaled = rescale_intensity(normalized, out_range=(0,255))
    masked = np.where(rescaled==0, np.nan, rescaled)
    return masked
            
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

def img_to_png(tif_path, png_path):
    """
    Converts processed tif chip images into pngs
    """
    arr = imread(tif_path)
    
    arr = np.dstack([arr[:,:,2],arr[:,:,1],arr[:,:,0]])

    imsave(png_path, img_as_ubyte(exposure.equalize_adapthist(arr)))
    
    
def label_to_png(tif_path, png_path):
    """
    Converts processed tif chip labels into pngs
    """
    arr = imread(tif_path)
    
    if len(arr.shape) > 2:
        arr = np.any(arr, axis=2)
                
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(png_path, img_as_ubyte(1*arr))
        
def imgs_to_pngs():
    """
    Extracts individual instances into their own tif files. Saves them
    in each folder ID in train folder. If an image has no instances,
    saves it with a empty mask.
    """
    for tif_path in self.raster_tile_paths:
        # for imgs with no instances, creates empty mask
        # only runs connected comp if there is at least one instance
        png_path = os.path.splitext(tif_path)[0] + ".png"
        img_to_png(tif_path, png_path)
    
def labels_to_pngs():
    """
    Extracts individual instances into their own tif files. Saves them
    in each folder ID in train folder. If an image has no instances,
    saves it with a empty mask.
    """
    for tif_path in self.rasterized_label_paths:
        # for imgs with no instances, creates empty mask
        # only runs connected comp if there is at least one instance
        png_path = os.path.splitext(tif_path)[0] + ".png"
        label_to_png(tif_path, png_path)     

        
def rename_imgs_anns_to_fcis_format(imfolder, annf):
    """
    FCIS has this goofy expectation that all images are named like this: COCO_splittype-name-number.jpg
    So we crawl the json and img files and rename them. it's requried to append "det" to the isntances 
    name in the previous step so that the name can be edited when saving the json used by fcis.
    """
    with open(annf, "r") as read_file:
        ann_json = json.load(read_file)
    for idx, i in enumerate(ann_json['images']):
        new_filename = f"COCO_{os.path.basename(imfolder)}_{str(idx+1).zfill(12)}.jpg"
        oldname = ann_json['images'][idx]['file_name']
        os.rename(os.path.join(imfolder,oldname), os.path.join(imfolder,new_filename))
        ann_json['images'][idx]['file_name'] = new_filename
    outname = os.path.join(os.path.dirname(annf), os.path.basename(annf)[4:])
    save_coco_annotation(outname, ann_json)
    return outname

def copy_imgs_to_fcis_structure(src_imgd, src_tuple, fcis_imgd, annf):
    """
    copies imgs to fcis folder and renames to FCIS COCO format.
    returns the annotation filename that has been udpated with new img paths
    """
    split_name = Path(src_tuple[0]).name.split(".")[0].split("_")[2]
    im_paths = [i for i in src_tuple[1]['jpeg_tiles'].to_list() if str(i).endswith(".jpg")]
    out_folder_name = Path(fcis_imgd)/ Path(split_name+"-nebraska")
    if os.path.exists(out_folder_name) is False:
        os.makedirs(str(out_folder_name))
    if len(os.listdir(str(out_folder_name))) ==0:
        assert len(os.listdir(str(out_folder_name))) !=1 # for .ipynb
        for i in im_paths:
            shutil.copy(str(i), str(out_folder_name / Path(i).name))
        outname = rename_imgs_anns_to_fcis_format(str(out_folder_name), annf)
        print(f"{outname} edited and saved.")
        return outname
    else:
        print(f"directory {out_folder_name} with imgs already exists")
        
def copy_annotations_to_fcis_structure(changed_name, src_tuple, fcis_annd):
    split_name = Path(src_tuple[0]).name.split(".")[0].split("_")[2]
    out_ann_name = Path(fcis_annd)/ Path("instances_"+split_name+"-nebraska.json")
    if out_ann_name.exists() is False:
        shutil.copy(changed_name, str(out_ann_name))
    else:
        print("annotations have already been copied to : "+ str(out_ann_name))