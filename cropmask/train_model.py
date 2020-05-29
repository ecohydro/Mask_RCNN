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
from detectron2.evaluation import COCOEvaluator
from cropmask.misc import make_dirs
from cropmask.coco_convert import split_save_train_validation_test_df, save_coco_annotation, create_coco_dataset
from cropmask import detectron2_reclass # fair amount of stuff goes on in here to make detectron work for this project.
from cropmask.detectron2_cropmask_cfg import cfg
from detectron2.engine import DefaultTrainer
# must be run after run_dask_preprocess.ipynb
def setup_register_load_inputs(cfg):
    tiles_path = Path(cfg.DATASET_PATH) / "tiles"
    
    coco_path = Path(cfg.DATASET_PATH) / "coco"
    train_coco_instances_path = str(coco_path / "det_instances_train.json")
    val_coco_instances_path = str(coco_path / "det_instances_val.json")
#     test_coco_instances_path = str(coco_path / "det_instances_test.json")
    # changes from images to jpeg
    # register each val and test set if there are more than one.
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, train_coco_instances_path, str(next(tiles_path.glob("*jpeg*"))))
#     register_coco_instances(cfg.DATASETS.VALIDATION[0], {}, val_coco_instances_path, str(next(tiles_path.glob("*image*"))))
    register_coco_instances(cfg.DATASETS.TEST[0], {}, val_coco_instances_path, str(next(tiles_path.glob("*jpeg*")))) # changed to val from test

def save_cfg(cfg):
    os.mkdir(cfg.OUTPUT_DIR)
    with open(Path(cfg.OUTPUT_DIR) / cfg.CONFIG_NAME, "w") as f: 
        f.write(cfg.dump()) 

def main():
    setup_register_load_inputs(cfg) # if this ain't here the multigpu can't find registered datasets
#     trainer = detectron2_reclass.Trainer(cfg) getting it to overfit first, no checking on validation set
    
    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    
    trainer = MyTrainer(cfg)
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
