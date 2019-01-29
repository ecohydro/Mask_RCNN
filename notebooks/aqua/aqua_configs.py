from mrcnn.config import Config
import numpy as np

############################################################
#  Configurations
############################################################

class AquaConfig(Config):
    """Configuration for training on Planet imagery. 
     Overrides values specific to PlanetScope Analytic SR scenes.
    
    Descriptive documentation for each attribute is at
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    
    There are many more hyperparameters to edit than are set in this subclass"""
    
    def __init__(self, N):
        """Set values of computed attributes. Channel dimension is overriden, 
        replaced 3 with N as per this guideline: https://github.com/matterport/Mask_RCNN/issues/314
        THERE MAY BE OTHER CODE CHANGES TO ACCOUNT FOR 3 vs N channels. See other 
        comments."""
        # https://github.com/matterport/Mask_RCNN/wiki helpful for N channels
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        
        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, N])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, N])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.CHANNELS_NUM = N
    
    LEARNING_RATE = .00002 
    
    # Image mean (RGBN RGBN) from WV2_MRCNN_PRE.ipynb
    # filling with N values, need to compute mean of each channel
    # values are for gridded wv2 no partial grids
#     MEAN_PIXEL = np.array([225.25, 308.74, 184.93])
    MEAN_PIXEL = np.array([756.72233265, 850.44464278, 740.32534265])
    
    # Give the configuration a recognizable name
    NAME = "aqua-planet"

    # Batch size is 4 (GPUs * images/GPU).
    # Keras 2.1.6 works for multi-gpu but takes longer than single GPU currently
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + lines + rafts + ponds + cages

    # Use small images for faster training. Determines the image shape.
    # From build() in model.py
    # Exception("Image size must be dividable by 2 at least 6 times "
    #     "to avoid fractions when downscaling and upscaling."
    #    "For example, use 256, 320, 384, 448, 512, ... etc. "
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # anchor side in pixels, determined using inspect_aqua_data.ipynb. can specify more or less scales
    RPN_ANCHOR_SCALES = (20, 60, 100, 140) # for aquaculture
    
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Aim to allow ROI sampling to pick 33% positive ROIs. This is always 33% in inspect_data nb, unsure if that is accurate.
    TRAIN_ROIS_PER_IMAGE = 100

    # Unsure what best step size is but nucleus used 100. Doubling because smallholder is more complex
    STEPS_PER_EPOCH = 100
    
    #reduces the max number of field instances
    MAX_GT_INSTANCES = 75 # for cp determined using inspect_aqua_data.ipynb

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Loss weights for more precise optimization. It has been suggested that mrcnn_mask_loss should be weighted higher
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    


class AquaInferenceConfig(AquaConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imagery for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7