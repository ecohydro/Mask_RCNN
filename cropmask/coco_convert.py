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
        if len(binary_mask.shape) > 2:
            for maskid in np.arange(binary_mask.shape[-1]):

                image_info = pycococreatortools.create_image_info(
                    image_id,os.path.basename(img_filename), (512,512))
                coco_output["images"].append(image_info)

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

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                (512,512), tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            image_id = image_id + 1
            
    return coco_output # josn of all the annotation and img path info for either a train/validate or test set

def save_coco_annotation(root_dir, output_fname, coco_output):
    with open(os.path.join(root_dir,"annotations", output_fname), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print(f"{output_fname}"+" saved.")
        
def copy_chips(matterport_chip_paths, coco_chip_paths):
    """
    Used for copying train or test chips to new folders for COCO.
    """
    for m_path, c_path in zip(matterport_chip_paths, coco_chip_paths):
        copyfile(m_path, c_path)
        
def copy_to_coco_folder_structure(ROOT_DIR='/mnt/cropmaskperm/test-landsat/', CHIP_DIR='/mnt/cropmaskperm/test-landsat/chips'):
    print("starting to copy file sto coco structure")
    misc.make_dirs([ROOT_DIR+"train", ROOT_DIR+"test", ROOT_DIR+"annotations"])
    label_train_validate_paths, label_test_paths, new_train_validate_paths, new_test_paths, old_train_validate_paths, old_test_paths = misc.train_test_split(CHIP_DIR, ROOT_DIR, 43, .1)
    copy_chips(old_test_paths, new_test_paths)
    copy_chips(old_train_validate_paths, new_train_validate_paths)
    print("done copying files")
    return label_train_validate_paths, label_test_paths, new_train_validate_paths, new_test_paths

def make_save_coco_json(ROOT_DIR):
    print("copying and saving coco json")
    CHIP_DIR=os.path.join(ROOT_DIR, chips)
    label_train_validate_paths, label_test_paths, new_train_validate_paths, new_test_paths = copy_to_coco_folder_structure(ROOT_DIR, CHIP_DIR)
    print("starting to save coco json")
    coco_meta_train = create_coco_meta("train")
    coco_meta_test = create_coco_meta("test")
    coco_output_train = create_coco_json(new_train_validate_paths, label_train_validate_paths, coco_meta_train)
    coco_output_test = create_coco_json(new_test_paths, label_test_paths, coco_meta_test)
    save_coco_annotation(ROOT_DIR, "instances_train.json", coco_output_train)
    save_coco_annotation(ROOT_DIR, "instances_test.json", coco_output_test)
    print("done saving coco json")