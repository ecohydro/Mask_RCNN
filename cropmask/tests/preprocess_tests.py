import cropmask.preprocess as pp
from cropmask.misc import make_dirs, remove_dirs
import os
import pytest

@pytest.fixture
def wflow():
    wflow = pp.PreprocessWorkflow("/home/ryan/work/CropMask_RCNN/cropmask/test_preprocess_config.yaml", 
                                 "/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050320312005082801T1-SC20190418222350/",
                                 "/permmnt/cropmaskperm/external/nebraska_pivots_projected.geojson")
    return wflow

def test_sequential_grid():
    from cropmask import sequential_grid as sg
    from rasterio import windows
    from cropmask.label_prep import rio_bbox_to_polygon
    from rasterio import coords
    from cropmask.io_utils import zipped_shp_url_to_gdf 
    import us
    import rasterio as rio
    import geopandas as gpd
    import matplotlib.pyplot as plt

    #specify zipped shapefile url
    nebraska_url = us.states.NE.shapefile_urls('state')
    gdf = zipped_shp_url_to_gdf(nebraska_url)
    band = rio.open("/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050280322005012001T2-SC20190818204900/LT05_L1GS_028032_20050120_20160912_01_T2_sr_band3.tif")
    gdf = gdf.to_crs(band.meta['crs'].to_dict())
    chip_list_full = sg.get_tiles_for_threaded_map(band, gdf, 512, 512)
    assert len(chip_list_full)==99
    # should also test for the label image

def test_init(wflow):
    
    assert wflow
    
def test_make_dir():
    
    directory_list = ["/permmnt/cropmaskperm/pytest_dir"]
    make_dirs(directory_list)
    try: 
        assert os.path.exists(directory_list[0])
    except AssertionError: 
        remove_dirs(directory_list)
        print("The directory was not created.")
    remove_dirs(directory_list)

def test_make_dirs(wflow):
    
    directory_list = wflow.setup_dirs()
    
    for i in directory_list:
        try: 
            assert os.path.exists(i)
        except AssertionError:
            remove_dirs(directory_list)
            print("The directory "+i+" was not created.")
    
    remove_dirs(directory_list)
    
def test_yaml_to_band_index(wflow):

    band_list = wflow.yaml_to_band_index()
    try: 
        assert band_list == ['1','2','3']
    except AssertionError:
        print("The band list "+band_list+" is not "+['1','2','3'])
        
def test_list_products():
    
    path = "/permmnt/cropmaskperm/unpacked_landsat_downloads/LT050320312005082801T1-SC20190418222350/"
    
    try: 
        product_list = os.listdir(path)
        assert product_list
    except AssertionError:
        print("The product list is empty, check this path: "+ path)
    
def test_get_product_paths(wflow):
   
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    assert product_list
    assert len(product_list) == len(band_list)
    
def test_stack_and_save_bands(wflow):
    
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    meta, bounds = wflow.load_meta_and_bounds(product_list)
    
    try: 
        wflow.stack_and_save_bands()
    except:
        remove_dirs(directory_list)
        print("The function didn't complete.")
    
    try: 
        assert os.path.exists(wflow.scene_path)
        remove_dirs(directory_list)
    except AssertionError:
        remove_dirs(directory_list)
        print("The stacked tif was not saved at the location "+wflow.scene_path)

def test_negative_buffer_and_small_filter(wflow):
    
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    meta, bounds = wflow.load_meta_and_bounds(product_list)
    
    wflow.stack_and_save_bands()
    
    try: 
        assert wflow.negative_buffer_and_small_filter(-31, 100) == np.array([0, 1]) # for the single class case, where 1 are cp pixels
    except:
        remove_dirs(directory_list)
        print("The function didn't complete.")
    
    try: 
        assert os.path.exists(wflow.rasterized_label_path)
        remove_dirs(directory_list)
    except AssertionError:
        remove_dirs(directory_list)
        print("The rasterized label tif was not saved at the location "+wflow.rasterized_label_path)
        
def test_grid_images(wflow):
    
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    meta, bounds = wflow.load_meta_and_bounds(product_list)
    
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    try: 
        img_paths, label_paths = wflow.grid_images()
        assert len(img_paths) > 0
        assert len(img_paths) == len(label_paths)
    except AssertionError:
        remove_dirs(directory_list)
        print("Less than one chip was saved") 
        
def test_move_chips_to_folder(wflow):
        
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    meta, bounds = wflow.load_meta_and_bounds(product_list)
        
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    
    img_paths, label_paths = wflow.grid_images()
    try: 
        assert wflow.move_chips_to_folder()
        assert len(os.listdir(wflow.TRAIN)) > 1
        assert len(os.listdir(os.listdir(wflow.TRAIN))[0]) > 0
    except AssertionError:
        remove_dirs(directory_list)
        print("Less than one chip directory was made") 
        
def test_connected_components(wflow):
        
    directory_list = wflow.setup_dirs()
    
    band_list = wflow.yaml_to_band_index()
    
    product_list = wflow.get_product_paths(band_list)
    
    meta, bounds = wflow.load_meta_and_bounds(product_list)
        
    wflow.stack_and_save_bands()
    
    wflow.negative_buffer_and_small_filter(-31, 100)
    
    img_paths, label_paths = wflow.grid_images()
    
    wflow.move_chips_to_folder()
    
    try: 
        assert wflow.connected_components()
    except AssertionError:
        print("Connected components did not complete") 

        