import numpy as np
import torch
import torch.nn as nn
from PIL import Image

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


def load_model(model_path, device=device):
    print('pytorch_classifier.py: Loading model...')
    return


def classify(model, image_bytes):
    print("classifying")
    return
