from mrcnn import utils
import os
import pandas as pd
import skimage.io as skio
from skimage import img_as_float
from scipy import ndimage
import numpy as np
import aqua_preprocess as pp

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = pp.ROOT
# Results directory
# Save submission files and test/train split csvs here
RESULTS_DIR = pp.RESULTS
AQUA_CLASS = pp.AQUA_CLASS

############################################################
#  Dataset
############################################################

class AquaDataset(utils.Dataset):
    """Generates the Imagery dataset."""
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        Channels are ordered [B, G, R, NIR]. This is called by the 
        Keras data_generator function
        """
        # Load image
        image = skio.imread(self.image_info[image_id]['path'])
        
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
    
        assert image.ndim == 3
    
        return image
    
    def load_aqua(self, dataset_dir, subset):
        """Load a subset of the aquaculture dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load.
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have four classes.
        # Naming the dataset aqua, and the class agriculture
        self.add_class("aqua", 1, "line")
        self.add_class("aqua", 2, "raft")
        self.add_class("aqua", 3, "pond")
        self.add_class("aqua", 4, "cage")

        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        train_ids = pd.read_csv(os.path.join(pp.RESULTS, 'train_ids.csv'))
        train_list = list(train_ids['train'])
        test_ids = pd.read_csv(os.path.join(pp.RESULTS, 'test_ids.csv'))
        test_list = list(test_ids['test'])
        
        # Set image ids based on subset
        if subset == "test":
            image_ids = test_list
        else:
            image_ids = train_list
        
        # Add images
        for image_id in image_ids:
            self.add_image(
                "aqua",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "image/{}.tif".format(image_id)))
       
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        
        # Get class mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "class_masks")

        # Read class mask files from .png image
        mask = [] # array to store instance masks
        classes = [] # array to store instance class 
        
        # Loop over class mask files
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".tif"):                
                
                # Read class mask
                m = skio.imread(os.path.join(mask_dir, f)).astype(np.bool)
                
                # Get class of instance mask from file name
                instance_class = f.split('_')[5]            
                                
                # Label objects in mask and save labeled image as "labels" and number of objects as "nb"
                labels, nb = ndimage.label(m)
                
                # Loob over labeled objects and isolate instance mask
                for obj in range(1,nb+1):
                    # Make new object from labels
                    instance = labels * 1
                    # Replace all other objects with 0s
                    instance[labels!=obj] = 0
                    # Set object values to 1 (rather than object id number)
                    instance[labels==obj] = 1

                    # Add instance mask to mask array
                    mask.append(instance)
                    classes.append(AQUA_CLASS[instance_class])
                    assert instance.ndim == 2        
        
        # Check if mask is still empty and, if so, add a np array of zeros (FIX THIS TO MATCH IMAGE SIZE)
        if not mask:
            m = np.zeros([256, 256], dtype=np.bool)
            mask.append(m)
            
        mask = np.stack(mask, axis=-1).astype(np.bool)
        assert mask.ndim == 3
        
        # Convert classess to proper data type
        classes = np.array(classes, dtype = np.int32)

        # Return mask, and array of class IDs of each instance
        return mask, classes
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "field":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
