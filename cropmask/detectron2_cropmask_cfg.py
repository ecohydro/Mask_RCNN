from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.VALIDATION = ("validation_nirrg",) # added after subclassing Default Trainer above in order to plot validtion loss curves during training. need to do 3 way split and register validation
cfg.DATASETS.TRAIN = ("train_nirrg",)
cfg.DATASETS.TEST = ("test_nirrg",)

# https://github.com/facebookresearch/detectron2/blob/dfc678a0aa6aaaae4e925877fe9f653edf627c86/detectron2/config/defaults.py
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 24
cfg.SOLVER.BASE_LR = 0.0003  # pick a good LR
cfg.SOLVER.MAX_ITER = 4000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # this is the default
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class 
cfg.TEST.EVAL_PERIOD = 0 # the period in terms of steps to evaluate the model on the test set by computing AP statistics, but should be edited to run on val set.
cfg.SOLVER.CHECKPOINT_PERIOD = 800
cfg.VALIDATION_PERIOD = 20
cfg.OUTPUT_DIR = "/home/ryan/work/cropmask_experiments/test1"
cfg.CUDNN_BENCHMARK = False # might cause speed gains since all images are same size if set to True (default is False)
cfg.INPUT.FORMAT = "RGB"
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
cfg.MODEL.PIXEL_MEAN = [ 931.90757125, 1001.46930161, 2793.30379383]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
cfg.MODEL.PIXEL_STD = [262.82447442, 340.61644907, 559.11205354]