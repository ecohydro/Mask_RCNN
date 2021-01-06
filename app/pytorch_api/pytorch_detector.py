from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image

use_gpu = True
dtype = torch.float32

device = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
print('Using device: ', device)

def open_image_and_meta(image_bytes):
    """ Opens a byte string as a numpy array and geotiff metadata with rasterio
    Args:
        image_bytes: a geotiff in binary format read from the POST request's body
    Returns:
        a numpy array and rasterio metadata object
    """
    with MemoryFile(image_bytes) as memfile:
        with memfile.open() as src:
            meta = src.meta
            arr = reshape_as_image(src.read())
    return arr, meta

def load_and_edit_cfg_for_inference(cfg_path= "/app/pytorch_api/config.yaml",
                                   model_weights_path = "/app/pytorch_api/best-mrcnn-nebraska-model-rgb-jpeg-split-geo-nebraska-freeze0-withseed.pth"):
    cfg = get_cfg()    # obtain detectron2's default config
    cfg.CONFIG_NAME = '' # add new configs for your own custom components
    cfg.DATASET_PATH = ''
    cfg.merge_from_file(cfg_path)   # load values from a file
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    model_cfg = cfg.clone()
    return model_cfg

def load_model_for_inference(cfg):
    print('pytorch_classifier.py: Loading model...')
    model = build_model(cfg)  # returns a torch.nn.Module
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS); # ; suppresses long output
    return model

def run_model_single_image(request_bytes, model, cfg):
    img, meta = open_image_and_meta(request_bytes)
    rgb_img = img[:,:,::-1] # assumes image is in BGR order, puts it in RGB order since model expects RGB
    aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    with torch.no_grad():
        height, width = img.shape[:2]
        img = aug.get_transform(rgb_img).apply_image(rgb_img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        inputs = {"image": img, "height": height, "width": width}
        predictions = model([inputs])
    cpu_output = predictions[0]["instances"].to("cpu")
    return cpu_output, rgb_img
