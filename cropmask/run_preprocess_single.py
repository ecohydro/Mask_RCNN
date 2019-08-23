from cropmask.preprocess import *
import time

param_path = "/home/ryan/work/CropMask_RCNN/cropmask/preprocess_config.yaml"
wflow = PreprocessWorkflow(param_path, 
                        "/permmnt/cropmaskperm/unpacked_landsat_downloads/032031/LT050320312005082801T1-SC20190418222350/",
                        "/permmnt/cropmaskperm/external/nebraska_pivots_projected.geojson")

if __name__ == "__main__":
    setup_dirs(param_path)
    wflow.run_single_scene()
    wflow.train_test_split()
    print("channel means for "+ self.scene_path)
    means = []
    for i in band_list:
        mean = get_arr_channel_mean(wflow.TRAIN,int(i)-1)
        means.append(mean)
        print("Band index {} mean for COCO normalization: ".format(i), mean)
