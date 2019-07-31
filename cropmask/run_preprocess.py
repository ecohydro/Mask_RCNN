from cropmask.preprocess import *

wflow = PreprocessWorkflow("/home/ryan/work/CropMask_RCNN/cropmask/preprocess_config.yaml", 
                                 "/mnt/cropmaskperm/unpacked_landsat_downloads/LT050320312005082801T1-SC20190418222350/",
                                 "/mnt/cropmaskperm/external/nebraska_pivots_projected.geojson")

if __name__ == "__main__":
    
    wflow.run_single_scene()
