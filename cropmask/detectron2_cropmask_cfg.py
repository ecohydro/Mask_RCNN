from detectron2.config import get_cfg
from detectron2 import model_zoo

# config URLs


cfg = get_cfg()
cfg.DATASET_PATH = "" # needs to be set since loading from base config with added attrs
cfg.CONFIG_NAME = "config.yaml"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
# added after subclassing Default Trainer above in order to plot validtion loss curves during training
cfg.DATASETS.VALIDATION = ("validation",) 
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("test",)
cfg.VALIDATION_PERIOD = 20
cfg.merge_from_file("/home/ryan/CropMask_RCNN/base_config.yaml")
cfg.DATASET_PATH = "/datadrive/test-ard-june-sept-nirrg"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.OUTPUT_DIR = "/datadrive/cropmask_experiments/nirrg-something/" # always change this for each unique experiment. config file for each run is saved in a new directory

## generalized R-CNN-FPN config
# https://github.com/facebookresearch/detectron2/blob/master/configs/Base-RCNN-FPN.yaml
# MODEL:
#   META_ARCHITECTURE: "GeneralizedRCNN"
#   BACKBONE:
#     NAME: "build_resnet_fpn_backbone"
#   RESNETS:
#     OUT_FEATURES: ["res2", "res3", "res4", "res5"]
#   FPN:
#     IN_FEATURES: ["res2", "res3", "res4", "res5"]
#   ANCHOR_GENERATOR:
#     SIZES: [[32], [64], [128], [256], [512]]  # One size for each in feature map
#     ASPECT_RATIOS: [[0.5, 1.0, 2.0]]  # Three aspect ratios (same for all in feature maps)
#   RPN:
#     IN_FEATURES: ["p2", "p3", "p4", "p5", "p6"]
#     PRE_NMS_TOPK_TRAIN: 2000  # Per FPN level
#     PRE_NMS_TOPK_TEST: 1000  # Per FPN level
#     # Detectron1 uses 2000 proposals per-batch,
#     # (See "modeling/rpn/rpn_outputs.py" for details of this legacy issue)
#     # which is approximately 1000 proposals per-image since the default batch size for FPN is 2.
#     POST_NMS_TOPK_TRAIN: 1000
#     POST_NMS_TOPK_TEST: 1000
#   ROI_HEADS:
#     NAME: "StandardROIHeads"
#     IN_FEATURES: ["p2", "p3", "p4", "p5"]
#   ROI_BOX_HEAD:
#     NAME: "FastRCNNConvFCHead"
#     NUM_FC: 2
#     POOLER_RESOLUTION: 7
#   ROI_MASK_HEAD:
#     NAME: "MaskRCNNConvUpsampleHead"
#     NUM_CONV: 4
#     POOLER_RESOLUTION: 14
# DATASETS:
#   TRAIN: ("coco_2017_train",)
#   TEST: ("coco_2017_val",)
# SOLVER:
#   IMS_PER_BATCH: 16
#   BASE_LR: 0.02
#   STEPS: (60000, 80000)
#   MAX_ITER: 90000
# INPUT:
#   MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
# VERSION: 2