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
from shutil import copyfile
import pandas as pd
from solaris.data.coco import geojson2coco
from sklearn.model_selection import train_test_split
from detectron2.data.datasets import register_coco_instances, load_coco_json

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

def get_tb_substring(fname):
    fname = str(fname)
    return re.search(r'.+_(-*\d+_\d+)\..+', fname).group(1)

def get_date_substring(fname):
    fname = str(fname)
    return re.search(r'.+_(\d+_\d+)_C.+', fname).group(1)

def match_by_tb_date(jpeg_tiles, image_tiles, label_tiles, geojson_tiles):
    """
    Faster way that doesn't work for some reason. ids look unique but lists aren't sorted the same
    
    def get_tb_substring2(fname):
        fname = str(fname)
        return re.search(r'.+_-*(\d+_\d+)\..+', fname).group(1)
    def get_date_substring2(fname):
        fname = str(fname)
        return re.search(r'.+_(\d+_\d+)_.+', fname).group(1)

    def get_uid(x):
        tb1, tb2 = get_tb_substring2(x).split("_")
        d1, d2 = get_date_substring2(x).split("_")
        return int(tb1 +tb2 + d1 + d2)

    sg = sorted(geojson_lst, key=lambda x: get_uid(x)) # sort doesn't work for some reason
    sl = sorted(label_lst, key=lambda x: get_uid(x))
    """
    print("total list length: {len(tile_df)}")
    match_lst = []
    count = 0
    for i in label_tiles:
        tb = get_tb_substring(i)
        date = get_date_substring(i)
        for j in geojson_tiles:
            if tb in j and date in j:
                break
        for k in image_tiles:
            if tb in k and date in k:
                break
        for z in jpeg_tiles:
            if tb in z and date in z:
                break
        match_lst.append([z,k,i,j])
        count +=1
        if count % 1000 == 0:
            print(count)
    return match_lst

def make_train_validation_test_df(tiles_path, save_empty_tiles = False):
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
    jpeg_tiles_path = tiles_path / "jpeg_tiles"
    label_tiles_path = tiles_path / "label_tiles"
    geojson_tiles_path = tiles_path / "geojson_tiles"
    label_tiles = list(label_tiles_path.glob("*.tif"))
    image_tiles = list(image_tiles_path.glob("*.tif"))
    geojson_tiles = list(geojson_tiles_path.glob("*.geojson"))
    jpeg_tiles = list(jpeg_tiles_path.glob("*.jpg"))
    assert len(label_tiles) > 0
    assert len(jpeg_tiles) > 0
    assert len(image_tiles) > 0
    # build tuples of label and im paths
    sorted_image_tiles = [str(i) for i in sorted(image_tiles, key=lambda x: str(x)[-19:])]
    sorted_jpeg_tiles = [str(i) for i in sorted(jpeg_tiles, key=lambda x: str(x)[-19:])]
    sorted_label_tiles = [str(i) for i in sorted(label_tiles, key=lambda x: str(x)[-19:])]
    sorted_geojson_tiles = [str(i) for i in sorted(geojson_tiles, key=lambda x: str(x)[-23:])]

    print("sort keys, these should follow same format and match")
    print(str(sorted_image_tiles[-1])[-19:])
    print(str(sorted_jpeg_tiles[-1])[-19:])
    print(str(sorted_label_tiles[-1])[-19:])
    print(str(sorted_geojson_tiles[-1])[-23:])
    match_lst = match_by_tb_date(sorted_jpeg_tiles, sorted_image_tiles, sorted_label_tiles, sorted_geojson_tiles)
    all_tiles_df = pd.DataFrame(match_lst, columns = ["jpeg_tiles", "image_tiles", "label_tiles", "geojson_tiles"])

    all_tiles_df['is_empty'] = all_tiles_df.loc[:,'label_tiles'].apply(str).str.contains("empty", regex=False)
    
    if save_empty_tiles is not True:
        all_tiles_df = all_tiles_df[all_tiles_df['is_empty']==False]
    all_tiles_df['tile_bounds'] = all_tiles_df['label_tiles'].apply(lambda x: get_tb_substring(x))
    all_tiles_df['date'] = all_tiles_df['label_tiles'].apply(lambda x: get_date_substring(x))

    return all_tiles_df


def split_save_train_validation_test_df(all_tiles_df, validation_size = .15, test_size = .05):
    not_test, test = train_test_split(np.unique(all_tiles_df['tile_bounds']), test_size=test_size, random_state=1)
    m = all_tiles_df.tile_bounds.isin(test)
    testdf = all_tiles_df[m]
    train, validation = train_test_split(not_test, test_size=validation_size, random_state = 1)
    m = all_tiles_df.tile_bounds.isin(train)
    traindf = all_tiles_df[m]
    vdf = all_tiles_df[~m]
    traindf.to_csv(tiles_path.parent / "coco" / "train.csv")
    validationdf.to_csv(tiles_path.parent / "coco" / "validation.csv")
    testdf.to_csv(tiles_path.parent / "coco" / "test.csv")
    return traindf, vdf, testdf
    

def create_coco_dataset(df):
    """
    Takes a df with paths to images and geojson tiles and creates coco formatted json for training or testing.
    """
    img_lst = df['image_tiles'].to_list()
    geojson_lst = df['geojson_tiles'].to_list()
    info, license, preset_categories = create_coco_meta() # preset cats unused for now, unsure how to properly work this with detectron
    # crazy regex is based on appending scene ID to each tile, including path/row and date info
    coco_dict = geojson2coco(image_src = [str(i) for i in img_lst],
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

def switch_tif_jpeg(tif_json):
    json_jpeg = []
    for idx, i in enumerate(tif_json['images']):
        tif_json['images'][idx]['file_name'] = tif_json['images'][idx]['file_name'].replace(".tif", ".jpg")
    return tif_json

def dataset_to_coco(dataset_path, img_type, experiment_dir=False):
    """
    Experiment dir only required for detectron2 dataset workflow. 
    For denmark dataset workflow, experiment dir is created seperately since it has it's own folder structure. And requires jpeg instead of tif.
    
    Randomly splits the dataset into train, validation, test by unique tile geographies.
    """
    tiles_path = Path(dataset_path) / "tiles"
    tile_df =  make_train_validation_test_df(tiles_path = tiles_path, save_empty_tiles = False)
    train, validation, test = split_save_train_validation_test_df(tile_df)
    coco_path = Path(dataset_path) / "coco"
    train_coco_instances_path = str(coco_path / "det_instances_train.json")
    val_coco_instances_path = str(coco_path / "det_instances_val.json")
    test_coco_instances_path = str(coco_path / "det_instances_test.json")
    if (coco_path / "instances_train.json").exists() is False:
        train_coco_dict = create_coco_dataset(train)
        val_coco_dict = create_coco_dataset(validation)
        test_coco_dict = create_coco_dataset(test)
        
        if img_type == "jpeg":
            train_coco_dict = switch_tif_jpeg(train_coco_dict)
            val_coco_dict = switch_tif_jpeg(val_coco_dict)
            test_coco_dict = switch_tif_jpeg(test_coco_dict)
                
        save_coco_annotation(train_coco_instances_path, train_coco_dict)
        save_coco_annotation(val_coco_instances_path, val_coco_dict)
        save_coco_annotation(test_coco_instances_path, test_coco_dict)
            
    else:
        print("COCO datasets already exist. Registering.")

    # register each val and test set if there are more than one.
    register_coco_instances("train", {}, train_coco_instances_path, str(next(tiles_path.glob("*jpeg*"))))
    register_coco_instances("validate", {}, val_coco_instances_path, str(next(tiles_path.glob("*jpeg*"))))
    register_coco_instances("test", {}, test_coco_instances_path, str(next(tiles_path.glob("*jpeg*"))))
    
    try:
        os.makedirs(experiment_dir, exist_ok=False)
    except:
        pass
    return (train_coco_instances_path, train), (val_coco_instances_path, validation), (test_coco_instances_path, test)
