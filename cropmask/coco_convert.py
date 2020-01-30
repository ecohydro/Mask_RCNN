from pathlib import Path
import skimage.io as skio
import numpy as np
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
from pycococreatortools import pycococreatortools
import cropmask.misc as misc
from shutil import copyfile
import pandas as pd
import solaris as sol
from sklearn.model_selection import train_test_split

# funcs derived from https://github.com/waspinator/pycococreator/blob/d29534e36aad6c30d7e4dadd9f4f7b0e344a774c/pycococreatortools/pycococreatortools.py
# and https://patrickwasp.com/create-your-own-coco-style-dataset/

def create_coco_meta():
    INFO_DICT = {
        "description": "2005 nebraska center pivots derived from CALMIT dataset",
        "url": "https://calmit.unl.edu/center-pivots-esri-shapefile-metadata-2005",
        "version": "0.1.0",
        "year": 2005,
        "contributor": "rbavery",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSE_DICT = {
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }

    PRESET_CATEGORIES = [
        {
            "id": 1,
            'name': 'center pivot',
            'supercategory': 'agriculture',
        },
    ]
    return INFO_DICT, LICENSE_DICT, PRESET_CATEGORIES
        
def split_save_train_validation_test_df(tiles_path, validation_size = .1, test_size = .1, random_state = 1, save_empty_tiles = False):
    """
    Takes a Path to tiles and randomly splits image and labels into train and test sets.
    The test split occurs first. After, the set that is not train is randomly split into 
    train and validation based on the validation size.
    
    Train and test csvs are saved in the coco folder that contains info on which tiles are
    empty. The df used to save these is also used by the path_df_to_coco_json function to 
    create a coco dataset in json format in the coco folder (can be used for train, validation, 
    and test df.)
    """
    image_tiles_path = tiles_path / "image_tiles"
    label_tiles_path = tiles_path / "label_tiles"
    geojson_tiles_path = tiles_path / "geojson_tiles"
    label_tiles = list(label_tiles_path.glob("*"))
    image_tiles = list(image_tiles_path.glob("*"))
    geojson_tiles = list(geojson_tiles_path.glob("*"))

    # build tuples of label and im paths
    sorted_image_tiles = sorted(image_tiles, key=lambda x: str(x)[-19:])
    sorted_label_tiles = sorted(label_tiles, key=lambda x: str(x)[-19:])
    sorted_geojson_tiles = sorted(geojson_tiles, key=lambda x: str(x)[-23:])

    all_tiles_df = pd.DataFrame(list(zip(sorted_image_tiles, sorted_label_tiles, sorted_geojson_tiles)), columns = ["image_tiles", "label_tiles", "geojson_tiles"])

    all_tiles_df['is_empty'] = all_tiles_df.loc[:,'label_tiles'].apply(str).str.contains("empty", regex=False)
    
    if save_empty_tiles is not True:
        all_tiles_df = all_tiles_df[all_tiles_df['is_empty']==False]

    not_test, test = train_test_split(all_tiles_df, test_size=test_size, random_state = 1)
    train, validation = train_test_split(not_test, test_size=validation_size, random_state = 1)

    train.to_csv(tiles_path.parent / "coco" / "train.csv")
    validation.to_csv(tiles_path.parent / "coco" / "validation.csv")
    test.to_csv(tiles_path.parent / "coco" / "test.csv")
    return train, validation, test

def create_coco_dataset(df):
    """
    Takes a df with paths to images and geojson tiles and creates coco formatted json for training or testing.
    """
    img_lst = df['image_tiles'].to_list()
    geojson_lst = df['geojson_tiles'].to_list()
    info, license, preset_categories = create_coco_meta() # preset cats unused for now, unsure how to properly work this with detectron
    # crazy regex is based on appending scene ID to each tile, including path/row and date info
    coco_dict = sol.data.coco.geojson2coco(image_src = [str(i) for i in img_lst],
                                       label_src = [str(i) for i in geojson_lst],
                                       matching_re=r'(\d{6}_\d{8}_\d{8}_C\d{2}_V\d_-?\d+_\d+)',
                                       remove_all_multipolygons = True,
                                       info_dict = info,
                                       license_dict = license,
                                       override_crs=True,
                                       verbose=0)
    return coco_dict

def save_coco_annotation(outpath, coco_output):
    with open(outpath, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print(f"{os.path.basename(outpath)}"+" saved.")
    
    
