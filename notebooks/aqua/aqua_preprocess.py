import random
import os
import shutil
import copy
from skimage import measure
from skimage import morphology as skim
import skimage.io as skio
import warnings
import pandas as pd
import numpy as np
import pathlib
import yaml
import geopandas as gpd
from rasterio import features, coords
import rasterio
from shapely.geometry import shape
import gdal

random.seed(1)

# Directory setup
ROOT = "/home/tclavelle/tana-crunch/CropMask_RCNN"
DATASET = os.path.join(ROOT, 'data/aqua')
PREPPED = os.path.join(DATASET, 'gridded_planet')
TRAIN = os.path.join(DATASET, 'train')
TEST = os.path.join(DATASET, 'test')
RESULTS = os.path.join(DATASET, 'results')

# Class dictionary
AQUA_CLASS = {"line": 1, "raft": 2, "pond": 3, "cage": 4}        

# Preprocessing functions
def make_dirs():
    
    dirs = [DATASET, PREPPED, TRAIN, TEST, RESULTS]

    # Make directory and subdirectories
    for d in dirs:
        pathlib.Path(d).mkdir(parents=False, exist_ok=False)

def remove_dir_folders(directory):
    """
    Removes old test/train directories before a new batch of training data is created
    """
    folderlist = [ f for f in os.listdir(directory)]
    for f in folderlist:
        if not f.startswith('.'):
            shutil.rmtree(os.path.join(directory,f))

def imgs_without_objects(directory):

    # Get directories inside prepped folder
    images = os.listdir(directory)
    images.remove('.DS_Store')

    # List to store files with no instances
    no_objects = []

    # For each file, check if any class mask file exists
    for i in images:

        # get list of class masks
        masks = os.listdir(os.path.join(pp.PREPPED, i,'class_masks'))

        # Empty vector of instances
        instances = []

        # Loop over masks and calculate instances
        for m in masks:

            arr = skio.imread(os.path.join(os.path.join(pp.PREPPED, i,'class_masks', m)))
            blob_labels = measure.label(arr, background=0)
            blob_vals = np.sum(np.unique(blob_labels))
            instances.append(blob_vals)

        # Find total number of instances
        if np.sum(instances) == 0:
            no_objects.append(i)

    return no_objects

def train_test_split(prepped_dir, train_dir, test_dir, kprop):
    """Takes a sample of folder ids and copies them to a test directory. 
    each sample folder containes an images and corresponding masks folder"""
    # Remove previous training and testing data from folders
    remove_dir_folders(test_dir)
    remove_dir_folders(train_dir)

    # List of available data samples
    sample_list = next(os.walk(prepped_dir))[1]
    
    # Add training samples to training folder
    for train_sample in sample_list:
        shutil.copytree(os.path.join(prepped_dir,train_sample),os.path.join(train_dir,train_sample))

    # Sample
    k = round(kprop*len(sample_list))
    test_list = random.sample(sample_list,k)

    # Add test sample to test folder
    for test_sample in test_list:
        shutil.copytree(os.path.join(prepped_dir,test_sample),os.path.join(test_dir,test_sample))
    
    train_list = list(set(next(os.walk(train_dir))[1]) - set(test_list))
    
    # Save csvs of result files
    train_df = pd.DataFrame({'train': train_list})
    test_df = pd.DataFrame({'test': test_list})
    train_df.to_csv(os.path.join(RESULTS, 'train_ids.csv'))
    test_df.to_csv(os.path.join(RESULTS, 'test_ids.csv'))
 
if __name__ == '__main__':
    print('sourced preprocess module')