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

from cropmask.misc import parse_yaml, make_dirs
from cropmask.coco_convert import split_save_train_validation_test_df, save_coco_annotation, create_coco_dataset
from cropmask import detectron2_reclass # fair amount of stuff goes on in here to make detectron work for this project.

def setup_register_load_inputs(cfg):
    param_path = "/home/ryan/work/CropMask_RCNN/test_inspection_config.yaml"
    params = parse_yaml(param_path)
    tiles_path = Path(os.path.join(params['dirs']['root'], params['dirs']['dataset'], "tiles"))

    train, validation, test = split_save_train_validation_test_df(tiles_path, save_empty_tiles=False)
    coco_path = Path(params['dirs']['root']) / params['dirs']['dataset'] / "coco"

    train_coco_instances_path = str(coco_path / "instances_train.json")
    val_coco_instances_path = str(coco_path / "instances_val.json")
    test_coco_instances_path = str(coco_path / "instances_test.json")
    if (coco_path / "instances_train.json").exists() is False:
        train_coco_dict = create_coco_dataset(train)
        val_coco_dict = create_coco_dataset(validation)
        test_coco_dict = create_coco_dataset(test)
        save_coco_annotation(train_coco_instances_path, train_coco_dict)
        save_coco_annotation(val_coco_instances_path, val_coco_dict)
        save_coco_annotation(test_coco_instances_path, test_coco_dict)
    # register each val and test set if there are more than one.
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, train_coco_instances_path, str(next(tiles_path.glob("*image*"))))
    register_coco_instances(cfg.DATASETS.VALIDATION[0], {}, val_coco_instances_path, str(next(tiles_path.glob("*image*"))))
    register_coco_instances(cfg.DATASETS.TEST[0], {}, test_coco_instances_path, str(next(tiles_path.glob("*image*"))))

    train_json = load_coco_json(train_coco_instances_path,  str(next(tiles_path.glob("*image*"))))
    val_json = load_coco_json(val_coco_instances_path,  str(next(tiles_path.glob("*image*"))))
    test_json = load_coco_json(test_coco_instances_path,  str(next(tiles_path.glob("*image*"))))

def main():
    from cropmask.detectron2_cropmask_cfg import cfg
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_register_load_inputs(cfg)
    trainer = detectron2_reclass.Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        dist_url="auto"
    )