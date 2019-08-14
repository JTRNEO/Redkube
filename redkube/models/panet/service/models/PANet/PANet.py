"""Perform inference on one or more datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import sys
import time
import subprocess
from collections import defaultdict
import pdb
# ROOT_DIR = os.path.join(os.getcwd(),'panet_utils')
# # print(ROOT_DIR)
# #ROOT_DIR = os.path.abspath("../")
import models.PANet.panet_utils.tools._init_paths
#import panet_utils.tools._init_paths  # pylint: disable=unused-import

import torch
import torch.nn as nn
from torch.autograd import Variable

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import pycocotools.mask as mask_util

import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

cfg.TEST.FORCE_JSON_DATASET_EVAL = 1


class PANet():
    def __init__(self,modelpath):
        self.model_path = modelpath

        dataset = "iiai"

        # Cfg
        #cfg_file="/home/ashwin/Desktop/panet/service/models/PANet/panet_utils/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml"
        #cfg_file="/home/ashwin/Desktop/panet/service/models/PANet/panet_utils/configs/panet/e2e_panet_R-101-FPN_2x_mask.yaml"
        cfg_file="/workspace/PANet/service/models/PANet/panet_utils/configs/panet/e2e_panet_R-152-FPN_2x_mask.yaml"
        #cfg_file="/media/ashwin/Apollo M100/MyFolders/Work/NinthFloor/PANet/panet/service/models/PANet/panet_utils/configs/panet/e2e_panet_R-152-FPN_2x_mask.yaml"
        
        # Ckpts
        #load_ckpt="/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res50_c28_step156650.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res50_c93_step239999.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res152_c28_step179999.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res152_c18_step179999.pth" #Last used
        load_ckpt = "/workspace/PANet/service/checkpoint/iiai/model_iiai_res152_c12_step179999.pth" #Last used
        #load_ckpt = "/media/ashwin/Apollo M100/MyFolders/Work/NinthFloor/PANet/panet/checkpoint/iiai/model_iiai_res152_c12_step179999.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res152_balanced_c12_step179999.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res101_c12_focal_step179999.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/spacenet/model_spacenet_step179999.pth"
        #load_ckpt = "/home/ashwin/Desktop/panet/checkpoint/iiai/model_iiai_res152_focal_c12_step179999.pth" # NO, FOCAL LOSS hurts performance. The vanilla PANet works much better
        
        cuda = True
        load_detectron = False


        #print('Called with args:')
        #print(self.args)

        # cfg.TEST.BBOX_AUG.SCALES = (1200, 1200, 1000, 800, 600, 400,200,200,100,100,50,25)

        # print("TEST_SCALES=",cfg.TEST.BBOX_AUG.SCALES)

        if dataset.startswith("iiai"):
            self.dataset = datasets.get_iiai_dataset()
            cfg.MODEL.NUM_CLASSES = len(self.dataset.classes)#1 + num_classes
        elif dataset.startswith("spacenet"):
            self.dataset = datasets.get_spacenet_dataset()
            cfg.MODEL.NUM_CLASSES = len(self.dataset.classes)#1 + 1
        else:
            raise ValueError('Unexpected dataset name: {}'.format(dataset))

        print('load cfg from file: {}'.format(cfg_file))
        cfg_from_file(cfg_file)

        # if self.args.set_cfgs is not None:
        #     cfg_from_list(self.args.set_cfgs)

        assert bool(load_ckpt) ^ bool(load_detectron), \
            'Exactly one of --load_ckpt and --load_detectron should be specified.'
        cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
        assert_and_infer_cfg(make_immutable=False)

        maskRCNN = Generalized_RCNN()

        if cuda:
            maskRCNN.cuda()


        if load_ckpt:
            load_name = load_ckpt
            print("loading checkpoint: %s" % (load_name))
            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            net_utils.load_ckpt(maskRCNN, checkpoint['model'])

        # if self.args.load_detectron:
        #     print("loading detectron weights %s" % self.args.load_detectron)
        #     load_detectron_weight(maskRCNN, self.args.load_detectron)

        self.maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True, device_ids=[0])  # only support single GPU

        self.maskRCNN.eval()


    def infere(self, image, imageId=None, thresh=0.5, debug=False, pixel_size=(0.3,0.3)):
        assert image is not None

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(self.maskRCNN, image, timers=timers)

        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)#self.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
            return []

        if segms is not None:
            masks = mask_util.decode(segms)

        result = []

        #print("Number of boxes=",len(boxes))
        #print("Boxes",boxes)
        #print("Boxes Shape=",boxes.shape)
        
        #masks = masks.T
        #print("Mask Shape=",masks.shape)
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        #sorted_inds = areas

        #for i in range(len(sorted_inds))
        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < thresh:
                continue

            class_id = i
            label = self.dataset.classes[classes[i]]
            area, area_m2, perimeter, cv2Poly = self.getMaskInfo(masks[:, :, i], image.shape, pixel_size=pixel_size) #masks[i].T, kernel=(10, 10)

            if cv2Poly is None:
                #print("Warning: Object is recognized, but contour is empty!")
                continue

            verts = cv2Poly[:, 0, :]
            r = {'classId': class_id,
                 'score': score,
                 'label': label,
                 'area': area,
                 'area_m2':area_m2,
                 'perimetr': perimeter,
                 'verts': verts}

            if imageId is not None:
                r['objId'] = "{}_obj-{}".format(imageId, i)

            result.append(r)

        return result


    def getMaskInfo(self, img, kernel=(10, 10), pixel_size=(0.3,0.3)):


        #Define kernel
        kernel = np.ones(kernel, np.uint8)

        #Open to erode small patches
        #thresh = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #Close little holes # change img to thresh to go back
        #thresh = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel, iterations=4)

        #thresh=thresh.astype('uint8') # change img to thresh to go back
        #contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        maxArea = 0
        maxContour = None

        # Get largest area contour
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > maxArea:
                maxArea = a
                maxContour = cnt

        if maxContour is None: return [None, None, None, None]

        perimeter = cv2.arcLength(maxContour,True)


        # Get area in meters squared
        pix_sq = pixel_size[0] * pixel_size[1]
        maxArea_m2 = maxArea * pix_sq 
        maxArea_m2 = round(maxArea_m2, 3)

        # aproximate contour with the 1% of squared perimiter accuracy
        # approx = cv2.approxPolyDP(maxContour, 0.01*math.sqrt(perimeter), True)


        return maxArea, maxArea_m2, perimeter, maxContour




































