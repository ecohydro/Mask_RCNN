from dataset.pycocotools.coco import COCO
from dataset.pycocotools import mask as maskUtils
import numpy as np
import cv2
import os
import colorsys
import random
import skimage
from skimage import morphology
from skimage.measure import find_contours
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import pandas as pd
    
# --------------------------------------------------------
# adaptd from (RA)
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Haochen Zhang, Yi Li, Haozhi Qi
# --------------------------------------------------------

def rle_to_mask_arrs(coco_ds, index):
    masks= []
    for key, subdict in coco_ds.anns.items():
        if subdict['image_id'] == index:
            segmask = coco_ds.anns[key]['segmentation']
            masks.append(segmask)
    return masks
    

def load_coco_annotation(index, anns_path, img_folder, num_classes=1):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        coco = COCO(anns_path)
        im_ann = coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = 0 # we only have 1 class
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if obj['iscrowd']:
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        roi_rec = {'image': os.path.join(img_folder, coco.loadImgs(index)[0]['file_name']),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'masks':rle_to_mask_arrs(coco, index),
                   'flipped': False}
        return roi_rec

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

#functions below come from: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
# They are copied as is or minimally edited so that mask shapes are like (512, 512, 100) and functions work with python 2
"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_masks, pred_boxes, pred_masks, 
        pred_scores, gt_class_ids, pred_class_ids,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1).astype(np.float32)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = {}
    AP_strs = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        
        AP_str = "AP at IOU {:.2f}: {:.3f}".format(iou_threshold, ap)
        AP.update({iou_threshold:ap})
        AP_strs.append(AP_str)
        print(AP_str)
    mrAP = np.array(AP.values()).mean() # mean across IOU range of APs
    AP_str = "AP across IOUs {:.2f}-{:.2f}: {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], mrAP)
    AP_strs.append(AP_str)
    AP.update({"0.5-0.95":mrAP})
    print(AP_str)
    return AP, AP_strs

def compute_recall_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AR = {}
    AR_strs = []
    for iou_threshold in iou_thresholds:
        ar, pos_ids =\
            compute_recall(pred_mask, gt_mask, iou=iou_threshold)
        
        AR_str = "AR at IOU {:.2f}: {:.3f}".format(iou_threshold, ar)
        AR.update({iou_threshold:ar})
        AR_strs.append(AR_str)
        print(AR_str)
    mrAR = np.array(AR.values()).mean() # mean across IOU range of ARs
    AR_str = "AR across IOUs {:.2f}-{:.2f}: {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], mrAR)
    AR_strs.append(AR_str)
    AR.update({"0.5-0.95":mrAR})
    print(AR_str)
    return AR, AR_strs

def positive_ids_from_masks(pred_masks, gt_masks, iou):
    overlaps = compute_overlaps_masks(pred_masks, gt_masks) # changed to masks
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    return np.where(iou_max >= iou)[0]

def compute_recall(pred_masks, gt_masks, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.
    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps_masks(pred_masks, gt_masks) # changed to masks
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = positive_ids_from_masks(pred_masks, gt_masks, iou)
    matched_gt_boxes = iou_argmax[positive_ids]
    recall = len(set(matched_gt_boxes)) / float(gt_masks.shape[-1])
    return recall, positive_ids

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def compute_matches(gt_boxes, gt_masks, pred_boxes, pred_masks, 
                     pred_scores, gt_class_ids, pred_class_ids,
                    iou_threshold=0.5, score_threshold=0.9):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = np.array(pred_class_ids)[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps

def remove_empty_masks_boxes(masks, boxes):
    bool_has_cp = masks.any(axis=(0,1))
    if False in bool_has_cp:
        indices_to_remove = np.where(~bool_has_cp)[0]
        return np.delete(masks, indices_to_remove, axis=-1), np.delete(boxes, indices_to_remove, axis=0) 
    else:
        return masks, boxes

def munge_p(coco_anns, index, all_ims, all_dets, all_masks):
    pred_masks = np.moveaxis(all_masks[index][0],0,-1)
    num_instances = pred_masks.shape[-1]
    pred_boxes = all_dets[index][0]
    full_pred_masks = [resize_binarize_msk(pred_boxes[i], pred_masks[...,i], all_ims[index]) for i in range(num_instances)]
    maskspred = np.moveaxis(np.array(full_pred_masks),0,-1)
    maskspred, pred_boxes = remove_empty_masks_boxes(maskspred, pred_boxes)
    pred_class_ids = maskspred.any(axis=(0,1))*1
    return pred_boxes, pred_class_ids, pred_boxes, maskspred

def munge_g(coco_anns, index, all_ims, all_dets, all_masks):
    size = all_ims[index].shape[0]
    full_gt_masks = [rle_to_full_mask(mask, size) for mask in coco_anns['masks']]
    masksgt = np.moveaxis(np.array(full_gt_masks),0,-1)
    gt_class_ids = masksgt.any(axis=(0,1))*1
    return gt_class_ids, masksgt

def munge_predictions_gt(index, all_ims, all_dets, all_masks, coco_anns):
    """Gets preds, gt in format for AR and AP calculations.
    """
    pred_boxes, pred_class_ids, pred_boxes, maskspred = munge_p(coco_anns, index, all_ims, all_dets, all_masks)
    gt_class_ids, masksgt = munge_g(coco_anns, index, all_ims, all_dets, all_masks)
    positive_ids = positive_ids_from_masks(maskspred, masksgt, .5)
    return coco_anns['boxes'], gt_class_ids, masksgt, pred_boxes, pred_class_ids, pred_boxes[:,-1], maskspred, positive_ids

def resize_binarize_msk(bbox, mask, im):
    coords = bbox[0:4].astype(int)
    minx, miny, maxx, maxy = coords
    full_msk = np.zeros_like(im[...,0])
    resized = cv2.resize(mask, full_msk[miny:maxy+1, minx:maxx+1].T.shape)
    full_msk[miny:maxy+1, minx:maxx+1] = (resized >= .7) * 1
    return full_msk

def resize_binarize_pad(bbox, mask, masked_image):
    mask = resize_binarize_msk(bbox, mask, masked_image)
#             masked_image = apply_mask(masked_image, mask, color) unused, it shows masks do not have holes. outlines are cleaner
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    mask = padded_mask
    return mask

def rle_to_full_mask(mask, size):
    return maskUtils.decode(maskUtils.frPyObjects(mask, size, size))[...,0]
    
def display_instances(image, masks, boxes, mask_thresh=.7, ec="red", class_ids=None, class_names=None,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      detection_mask=True, show_bbox=True,
                      colors=None, captions=None, auto_show=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    title: (optional) Figure title
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    detection_mask: need to convert from rle if not
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title, fontsize=18)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
#         if not np.any(boxes[i]):
#             # Skip this instance. Has no bbox. Likely lost in image cropping.
#             continue
#         y1, x1, y2, x2 = boxes[i]
#         if show_bbox:
#             p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
#                                 alpha=0.7, linestyle="dashed",
#                                 edgecolor=color, facecolor='none')
#             ax.add_patch(p)

        # Label
        if captions:
            class_id = class_ids[i]
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
            caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        contours = find_contours(mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = PathPatch(Path(verts), fc = "none", edgecolor=ec, linewidth=2)
            ax.add_patch(p)
    if auto_show:
        ax.imshow(image)
    return ax

# Compute AP over range 0.5 to 0.95 and print it
def calc_stats_and_strs(gt_boxes, gt_class_ids, masksgt, pred_boxes, pred_class_ids, 
                             pred_scores, maskspred):
    aps, ap_strs = compute_ap_range(gt_box=gt_boxes, 
                             gt_class_id=gt_class_ids, gt_mask=masksgt, 
                             pred_box=pred_boxes, pred_class_id=pred_class_ids, 
                             pred_score=pred_scores, pred_mask=maskspred)

    ars, ar_strs = compute_recall_range(gt_box=gt_boxes, 
                             gt_class_id=gt_class_ids, gt_mask=masksgt, 
                             pred_box=pred_boxes, pred_class_id=pred_class_ids, 
                             pred_score=pred_scores, pred_mask=maskspred)

    return "\n".join(ap_strs), "\n".join(ar_strs), aps, ars


def stats_df(aps, ars):
    ap_df = pd.DataFrame.from_dict(aps, orient="index", columns=["AP"])
    map_s = ap_df.loc["0.5-0.95"].values
    ap_df_d = ap_df.drop("0.5-0.95")
    ar_df = pd.DataFrame.from_dict(ars, orient="index", columns=["AR"])
    mar_s = ar_df.loc["0.5-0.95"].values
    ar_df_d = ar_df.drop("0.5-0.95")
    ar_df_d.index = ar_df_d.index.astype(float)
    ar_df_d = ar_df_d.sort_index()
    ap_df_d.index = ap_df_d.index.astype(float)
    ap_df_d = ap_df_d.sort_index()
    ap_df_d["AR"] = ar_df_d["AR"]
    return ap_df_d, map_s, mar_s

def predict_on_image_names(image_names, config, model_path_id="/home/data/output/resnet_v1_101_coco_fcis_end2end_ohem-nebraska/train-nebraska/e2e",epoch=8):
    import argparse
    import os
    import sys
    import logging
    import pprint
    import cv2
    from utils.image import resize, transform
    import numpy as np
    # get config
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
    cur_path = os.path.abspath(".")
    sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
    import mxnet as mx
    print("use mxnet at", mx.__file__)
    from core.tester import im_detect, Predictor
    from symbols import *
    from utils.load_model import load_param
    from utils.show_masks import show_masks
    from utils.tictoc import tic, toc
    from nms.nms import py_nms_wrapper
    from mask.mask_transform import gpu_mask_voting, cpu_mask_voting
    # get symbol
    ctx_id = [int(i) for i in config.gpus.split(',')]
    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 2
    classes = ['cp']

    # load demo data
    data = []
    for im_name in image_names:
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

     # loading the last epoch that was trained, 8
    arg_params, aux_params = load_param(model_path_id, epoch, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(ctx_id[0])], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    all_classes = []
    all_configs = []
    all_masks = []
    all_dets = []
    all_ims = []

    # warm up
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales, config)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in xrange(num_classes)]
            all_masks = [[] for _ in xrange(num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, num_classes)]
            masks = [all_masks[j] for j in range(1, num_classes)]
        else:
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print (im_height, im_width)
            boxes = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, ctx_id[0])

            dets = [result_dets[j] for j in range(1, num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
        print ('testing {} {:.4f}s'.format(im_name, toc()))
        # visualize
        for i in xrange(len(dets)):
            keep = np.where(dets[i][:,-1]>0.7)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]

        all_classes.append(classes)
        all_configs.append(config)
        all_masks.append(masks)
        all_dets.append(dets)
        im = cv2.imread(im_name)
        all_ims.append(im)
    return all_ims, all_dets, all_masks, all_configs, all_classes