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
from sklearn.model_selection import train_test_split

# funcs derived from https://github.com/waspinator/pycococreator/blob/d29534e36aad6c30d7e4dadd9f4f7b0e344a774c/pycococreatortools/pycococreatortools.py
# and https://patrickwasp.com/create-your-own-coco-style-dataset/

def create_coco_meta(set_id):
    INFO = {
        "description": "nebraska center pivots "+set_id,
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2005,
        "contributor": "rbavery",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'agriculture',
            'supercategory': 'shape',
        },
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    return coco_output

def get_paths_from_preprocessed_chips(chips_dir, glob_pattern='**/*.tif'):
    """
    Recursively finds the mask and image tifs in folders created
    from PreprocessWorflow class. Source folders are of shape
    ../chips/landsat_id/image/tile.tif and are not compatible with COCO
    format. COCO format needed for tensorpack MaskRCNN.
    
    Arg:
        root_dir: path to chips folder
        glob_pattern: pattern to search for tif files, shouldn't need to change
            if chips folder is used.
    """
    img_files = []
    label_files = []
    for filename in Path(chips_dir).glob(glob_pattern):
        if 'label' not in filename.as_posix():
            img_files.append(filename)
        else:
            label_files.append(filename)
    return img_files, label_files



def create_coco_json(img_files, label_files, coco_output, fileext=".tif"):
    """
    Creates coco json from chips folder after Preprocess workflow is done.
    fileext argument is for creating json from tif (for training) or png (for)
    coco annotator. coco_output comes from coco_meta()
    """
    image_id = 1
    segmentation_id = 1 # every annotation needs to be unique across whole dataset
    for image_filename, label_filename in zip(img_files, label_files):
        img_filename = os.path.splitext(image_filename)[0]+fileext
        binary_mask = skio.imread(label_filename)
        
        image_info = pycococreatortools.create_image_info(image_id,
                os.path.basename(img_filename), (512,512))
        coco_output["images"].append(image_info)
                
        if len(binary_mask.shape) > 2:
            for maskid in np.arange(binary_mask.shape[-1]):

                category_info = {'id': 1, 'is_crowd': False}

                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask[:,:,maskid],
                    (512,512), tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1

            image_id = image_id + 1

        else:
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(img_filename), (512,512))
            coco_output["images"].append(image_info)

            category_info = {'id': 1, 'is_crowd': False}
            
            segmentation_id = segmentation_id + 1

            annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    (512,512), tolerance=2)

            if annotation_info is not None:
                
                coco_output["annotations"].append(annotation_info)

            image_id = image_id + 1
            
    return coco_output # josn of all the annotation and img path info for either a train/validate or test set

def save_coco_annotation(outpath, coco_output):
    with open(outpath, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print(f"{os.path.basename(outpath)}"+" saved.")
        
def copy_chips(matterport_chip_paths, coco_chip_paths):
    """
    Used for copying train or test chips to new folders for COCO.
    """
    for m_path, c_path in zip(matterport_chip_paths, coco_chip_paths):
        copyfile(m_path, c_path)
        
def split_save_train_test_df(tiles_path, save_empty_tiles = True):
    """
    Takes a Path to tiles and splits image and labels into train and test sets.
    
    Train and test csvs are saved in the coco folder that contains info on which tiles are
    empty. The df used to save these is also used by the path_df_to_coco_json function to 
    create a coco dataset in json format in the coco folder (can be used for both train 
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

    train, test = train_test_split(all_tiles_df, test_size=0.1, random_state = 1)

    train.to_csv(tiles_path.parent / "coco" / "train.csv")
    test.to_csv(tiles_path.parent / "coco" / "test.csv")
    return train, test

def path_df_to_coco_json(df, split_id, coco_path):
    print("creating coco json")
    coco_meta = create_coco_meta(split_id)
    coco_output = create_coco_json(df["image_tiles"].to_list(), df["label_tiles"].to_list(), coco_meta)
    outpath = os.path.join(coco_path, f"instances_{split_id}.json")
    save_coco_annotation(outpath, coco_output)
    print("done saving coco json")
    return outpath
