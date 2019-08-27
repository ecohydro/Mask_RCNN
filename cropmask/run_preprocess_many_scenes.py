from cropmask.preprocess import PreprocessWorkflow, get_arr_channel_mean, setup_dirs
import time
import dask

def run_scene_path_list(param_path, scene_path_list, labels_path):
    """Runs the preprocessing workflow on a list of scenes with dask delayed.
    Other params, such as the labels path, are configured in the preprocessing yaml file.
        
    Args:
        scene_path_list (list of strings): a list of absolute paths to the Landsat scenes
            
    Returns:
        The channel wise means for each Landsat band for the training data, in BGR order.
            
    """
    results = []
    for scene_path in scene_path_list:
        
        wflow = dask.delayed(PreprocessWorkflow)(param_path, scene_path, labels_path)
        
        result = dask.delayed(wflow.run_single_scene)()
        
        results.append(result)
    # https://docs.dask.org/en/stable/delayed-best-practices.html
    dask.compute(*results)
        
if __name__ == "__main__":
    
    start = time.time()
    param_path = "/home/ryan/work/CropMask_RCNN/cropmask/test_preprocess_config.yaml"
    scene_list = [
        "/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050280322005012001T2-SC20190818204900", 
        "/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050310322005021001T1-SC20190818205059",
        "/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050290312005031601T1-SC20190818204935",  
        "/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050320312005011601T1-SC20190818205113",
    ]
    labels_path = "/permmnt/cropmaskperm/external/nebraska_pivots_projected.geojson"
    
    setup_dirs(param_path)
    
    run_scene_path_list(param_path,
                       scene_list,
                       labels_path)
    
    # this is just to get the train dir path
    wflow = PreprocessWorkflow(param_path, 
                                 scene_list[0],
                                 labels_path)
    means = []
    for i in wflow.band_list:
        mean = get_arr_channel_mean(wflow.TRAIN,int(i)-1)
        means.append(mean)
        print("Band index {} mean for COCO normalization: ".format(i), mean)
        
    stop = time.time()
        
    print(stop-start, " seconds for this number of scenes: " + len(scene_list))
