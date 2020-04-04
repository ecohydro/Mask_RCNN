import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import some common libraries
from pathlib import Path
import os

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, launch, default_argument_parser
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from cropmask.misc import make_dirs
from cropmask.coco_convert import split_save_train_validation_test_df, save_coco_annotation, create_coco_dataset
from cropmask import detectron2_reclass # fair amount of stuff goes on in here to make detectron work for this project.
from cropmask.detectron2_cropmask_cfg_denmark import cfg

from detectron2.engine import DefaultTrainer

def setup_register_load_inputs(cfg):
    outpath = Path(r'/datadrive/denmark-data/preprocessed')
    im_tiles_path = outpath / r'images/train2016'
    val_tiles_path = outpath / r'images/val2016'
    train_coco_instances_path = outpath / r'annotations/instances_train2016.json'
    val_coco_instances_path = outpath / r'annotations/instances_val2016.json'
    
   # register each val and test set if there are more than one.
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, str(train_coco_instances_path), str(im_tiles_path))
    register_coco_instances(cfg.DATASETS.VALIDATION[0], {}, str(val_coco_instances_path), str(val_tiles_path))
#     register_coco_instances(cfg.DATASETS.TEST[0], {}, test_coco_instances_path, str(next(tiles_path.glob("*jpeg*"))))

    train_json = load_coco_json(str(train_coco_instances_path),  str(im_tiles_path))
    val_json = load_coco_json(str(val_coco_instances_path),  str(val_tiles_path))
#test_json = load_coco_json(test_coco_instances_path,  str(next(tiles_path.glob("*jpg*"))))

def save_cfg(cfg):
    os.mkdir(cfg.OUTPUT_DIR)
    with open(Path(cfg.OUTPUT_DIR) / cfg.CONFIG_NAME, "w") as f: 
        f.write(cfg.dump()) 

def main():
    setup_register_load_inputs(cfg) # if this ain't here the multigpu can't find registered datasets
#     trainer = detectron2_reclass.Trainer(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    save_cfg(cfg)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        dist_url="auto"
    )
