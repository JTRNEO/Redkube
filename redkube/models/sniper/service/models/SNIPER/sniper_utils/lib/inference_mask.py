# ------------------------------------------------------------------
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference module for performing detection and proposal extraction
# Written by Mahyar Najibi
# -----------------------------------------------------------------
import numpy as np
from bbox.bbox_transform import bbox_pred, clip_boxes
from iterators.PrefetchingIter import PrefetchingIter
import os
import time
import cPickle
from data_utils.data_workers import nms_worker, mask_nms_worker
from data_utils.visualization import visualize_dets
from data_utils.visualization import visualize_masks
from tqdm import tqdm
import math
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from iterators.MNIteratorTest import MNIteratorTest
import mxnet as mx
from nms.nms import py_nms_wrapper, gpu_nms_wrapper
import timeit


import pdb
# TODO: 1. finish single batch code (on demo.py already finished)
# TODO: 2. finish multi batch code(on main_test.py)
class Tester(object):
    def __init__(self, module, imdb, roidb, test_iter, cfg, rcnn_output_names=None, rpn_output_names=None,
                 logger=None, batch_size=None):
        self.test_iter = test_iter

        # Make sure that iterator is instnace of Prefetching iterator
        if test_iter and not isinstance(test_iter, PrefetchingIter):
            self.test_iter = PrefetchingIter(self.test_iter)
            self.scale = test_iter.test_scale
        self.cfg = cfg
        self.module = module
        if test_iter:
            self.data_names = [k[0] for k in test_iter.provide_data_single]
        self.rcnn_output_names = rcnn_output_names
        if not self.rcnn_output_names:
            self.rcnn_output_names = {
                'cls': 'cls_prob_reshape_output',
                'bbox': 'bbox_pred_reshape_output',
                'im_ids': 'im_ids',
                'mask': 'mask_prob_reshape_output'
            }
        self.rpn_output_names = rpn_output_names
        if not self.rpn_output_names:
            self.rpn_output_names = {
                'scores': 'rois_score',
                'rois': 'rois_output',
                'im_ids': 'im_ids'
            }
        self.logger = logger
        self.result_path = imdb.result_path
        self.num_classes = imdb.num_classes
        self.class_names = imdb.classes
        self.num_images = len(roidb)
        self.imdb_name = imdb.name
        # self.nms_worker = nms_worker(cfg.TEST.NMS, cfg.TEST.NMS_SIGMA)
        # TODO: check it
        self.nms_worker = mask_nms_worker(cfg.TEST.NMS, cfg.TEST.NMS_SIGMA)
        self.batch_size = batch_size
        self.roidb = roidb
        self.verbose = len(roidb) > 1
        self.thread_pool = None

        if not self.batch_size:
            self.batch_size = self.cfg.TEST.BATCH_IMAGES

    def forward(self, batch):
        self.module.forward(batch, is_train=False)
        return [dict(zip(self.module.output_names, i))
                for i in zip(*self.module.get_outputs(merge_multi_context=False))]

    def get_proposals(self, batch, scales):
        data = dict(zip(self.data_names, batch.data))
        outputs = self.forward(batch)
        scores, rois = [], []
        im_shapes = np.array([im.shape[-2:] for im in data['data']]).reshape(-1, self.batch_size, 2)
        im_ids = np.array([], dtype=int)
        for i, (gpu_out, gpu_scales, gpu_shapes) in enumerate(zip(outputs, scales, im_shapes)):

            gpu_rois = gpu_out[self.rpn_output_names['rois']].asnumpy()
            # Reshape crois
            nper_gpu = gpu_rois.shape[0] / self.batch_size
            gpu_scores = gpu_out[self.rpn_output_names['scores']].asnumpy()
            im_ids = np.hstack((im_ids, gpu_out[self.rpn_output_names['im_ids']].asnumpy().astype(int)))
            for idx in range(self.batch_size):
                cids = np.where(gpu_rois[:, 0] == idx)[0]
                assert len(cids) == nper_gpu, 'The number of rois per GPU should be fixed!'
                crois = gpu_rois[cids, 1:] / gpu_scales[idx]
                cscores = gpu_scores[cids]
                # Store predictions
                scores.append(cscores)
                rois.append(crois)
        return scores, rois, data, im_ids

    def detect(self, batch, scales):
        data = dict(zip(self.data_names, batch.data))
        outputs = self.forward(batch)
        scores, preds = [], []
        masks = []
        im_shapes = np.array([im.shape[-2:] for im in data['data']]).reshape(-1, self.batch_size, 2) # shape (1,batch_size,2)
        im_ids = np.array([], dtype=int)
        # at each img, the mask shape is [1000, 36, 28, 28] 36 = num_classes(18) * 2
        # print 'check the shape of mask here'
        # pdb.set_trace()
        for i, (gpu_out, gpu_scales, gpu_shapes) in enumerate(zip(outputs, scales, im_shapes)):
            gpu_rois = gpu_out[self.rpn_output_names['rois']].asnumpy()
            # Reshape crois
            nper_gpu = gpu_rois.shape[0] / self.batch_size
            gpu_scores = gpu_out[self.rcnn_output_names['cls']].asnumpy()
            gpu_deltas = gpu_out[self.rcnn_output_names['bbox']].asnumpy()
            # pdb.set_trace()
            # gpu_masks = gpu_out[self.rcnn_output_names['mask']].asnumpy()
            gpu_masks = gpu_out[self.rcnn_output_names['mask']].asnumpy()
            # pdb.set_trace()
            im_ids = np.hstack((im_ids, gpu_out[self.rcnn_output_names['im_ids']].asnumpy().astype(int)))
            # pdb.set_trace()

            for idx in range(self.batch_size):
                cids = np.where(gpu_rois[:, 0] == idx)[0]
                assert len(cids) == nper_gpu, 'The number of rois per GPU should be fixed!'
                crois = gpu_rois[cids, 1:]
                cscores = gpu_scores[idx]
                cdeltas = gpu_deltas[idx]
                cmasks = gpu_masks[idx]

                # Apply deltas and clip predictions
                cboxes = bbox_pred(crois, cdeltas)
                cboxes = clip_boxes(cboxes, gpu_shapes[idx])

                # Re-scale boxes
                cboxes = cboxes / gpu_scales[idx]

                # Store predictions
                scores.append(cscores)
                preds.append(cboxes)
                masks.append(cmasks)

        return scores, preds, data, im_ids, masks

    def set_scale(self, scale):
        if isinstance(self.test_iter, PrefetchingIter):
            self.test_iter.iters[0].set_scale(scale)
        else:
            self.test_iter.set_scale(scale)
        self.test_iter.reset()

    def show_info(self, print_str):
        print(print_str)
        if self.logger: self.logger.info(print_str)


    def aggregateSingleService(self, scale_cls_dets, scale_cls_masks, vis=False, cache_name='cache', vis_path=None, vis_name=None,
                  vis_ext='.png'):

        n_scales = len(scale_cls_dets)
        
        assert n_scales == len(self.cfg.TEST.VALID_RANGES), 'A valid range should be specified for each test scale'

        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        all_masks = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        if len(scale_cls_dets) > 1:
            self.show_info('Aggregating detections from multiple scales and applying NMS...')
        else:
            self.show_info('Performing NMS on detections...')

        # TODO: change the hard code here, change it to soft_nms or mask_nms
        nms = py_nms_wrapper(0.3)
        # nms = gpu_nms_wrapper(0.3, 0)
        # Apply ranges and store detections per category
        for i in range(self.num_images):
            for j in range(1, self.num_classes):
                agg_dets = np.empty((0,5),dtype=np.float32)
                agg_masks = np.empty((0, 28, 28), dtype=np.float32)
                for all_cls_dets, all_cls_masks, valid_range in zip(scale_cls_dets, scale_cls_masks, self.cfg.TEST.VALID_RANGES):
                    cls_dets = all_cls_dets[j][i]
                    cls_masks = all_cls_masks[j][i]
                    heights = cls_dets[:, 2] - cls_dets[:, 0]
                    widths = cls_dets[:, 3] - cls_dets[:, 1]
                    areas = widths * heights
                    lvalid_ids = np.where(areas > valid_range[0]*valid_range[0])[0] if valid_range[0] > 0 else \
                        np.arange(len(areas))
                    uvalid_ids = np.where(areas <= valid_range[1]*valid_range[1])[0] if valid_range[1] > 0 else \
                        np.arange(len(areas))
                    valid_ids = np.intersect1d(lvalid_ids,uvalid_ids)
                    cls_dets = cls_dets[valid_ids, :] if len(valid_ids) > 0 else cls_dets
                    cls_masks = cls_masks[valid_ids, :, :] if len(valid_ids) > 0 else cls_masks
                    # pdb.set_trace()
                    agg_dets = np.vstack((agg_dets, cls_dets.astype(np.float32)))
                    # pdb.set_trace()
                    agg_masks = np.concatenate((agg_masks, cls_masks), axis=0)
                # start = timeit.default_timer()
                keep = nms(agg_dets)
                # stop = timeit.default_timer()
                # print 'nms time: ', stop - start
                all_boxes[j][i] = agg_dets[keep, :]
                all_masks[j][i] = agg_masks[keep, :]
                # parallel_nms_args[int(i/n_roi_per_pool)].append(agg_dets)

        # Divide roidb and perform NMS in parallel to reduce the memory usage
        # TODO: change to multi process later

        # Limit number of detections to MAX_PER_IMAGE if requested and visualize if vis is True
        for i in range(self.num_images):
            if self.cfg.TEST.MAX_PER_IMAGE > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, self.num_classes)])
                if len(image_scores) > self.cfg.TEST.MAX_PER_IMAGE:
                    image_thresh = np.sort(image_scores)[-self.cfg.TEST.MAX_PER_IMAGE]
                    for j in range(1, self.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
                        all_masks[j][i] = all_masks[j][i][keep, :]

        if cache_name:
            cache_path = os.path.join(self.result_path, cache_name)
            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)
            cache_path = os.path.join(cache_path, 'detections.pkl')
            self.show_info('Done! Saving detections into: {}'.format(cache_path))
            with open(cache_path, 'wb') as detfile:
                cPickle.dump(all_boxes, detfile)
        return all_boxes, all_masks




    def aggregateSingle(self, scale_cls_dets, scale_cls_masks, vis=False, cache_name='cache', vis_path=None, vis_name=None,
                  vis_ext='.png'):

        n_scales = len(scale_cls_dets)
        assert n_scales == len(self.cfg.TEST.VALID_RANGES), 'A valid range should be specified for each test scale'

        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        all_masks = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        if len(scale_cls_dets) > 1:
            self.show_info('Aggregating detections from multiple scales and applying NMS...')
        else:
            self.show_info('Performing NMS on detections...')

        # TODO: change the hard code here, change it to soft_nms or mask_nms
        nms = py_nms_wrapper(0.3)
        # nms = gpu_nms_wrapper(0.3, 0)
        # Apply ranges and store detections per category
        for i in range(self.num_images):
            for j in range(1, self.num_classes):
                agg_dets = np.empty((0,5),dtype=np.float32)
                agg_masks = np.empty((0, 28, 28), dtype=np.float32)
                for all_cls_dets, all_cls_masks, valid_range in zip(scale_cls_dets, scale_cls_masks, self.cfg.TEST.VALID_RANGES):
                    cls_dets = all_cls_dets[j][i]
                    cls_masks = all_cls_masks[j][i]
                    heights = cls_dets[:, 2] - cls_dets[:, 0]
                    widths = cls_dets[:, 3] - cls_dets[:, 1]
                    areas = widths * heights
                    lvalid_ids = np.where(areas > valid_range[0]*valid_range[0])[0] if valid_range[0] > 0 else \
                        np.arange(len(areas))
                    uvalid_ids = np.where(areas <= valid_range[1]*valid_range[1])[0] if valid_range[1] > 0 else \
                        np.arange(len(areas))
                    valid_ids = np.intersect1d(lvalid_ids,uvalid_ids)
                    cls_dets = cls_dets[valid_ids, :] if len(valid_ids) > 0 else cls_dets
                    cls_masks = cls_masks[valid_ids, :, :] if len(valid_ids) > 0 else cls_masks
                    # pdb.set_trace()
                    agg_dets = np.vstack((agg_dets, cls_dets.astype(np.float32)))
                    # pdb.set_trace()
                    agg_masks = np.concatenate((agg_masks, cls_masks), axis=0)
                # start = timeit.default_timer()
                keep = nms(agg_dets)
                # stop = timeit.default_timer()
                # print 'nms time: ', stop - start
                all_boxes[j][i] = agg_dets[keep, :]
                all_masks[j][i] = agg_masks[keep, :]
                # parallel_nms_args[int(i/n_roi_per_pool)].append(agg_dets)

        # Divide roidb and perform NMS in parallel to reduce the memory usage
        # TODO: change to multi process later

        # Limit number of detections to MAX_PER_IMAGE if requested and visualize if vis is True
        for i in range(self.num_images):
            if self.cfg.TEST.MAX_PER_IMAGE > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, self.num_classes)])
                if len(image_scores) > self.cfg.TEST.MAX_PER_IMAGE:
                    image_thresh = np.sort(image_scores)[-self.cfg.TEST.MAX_PER_IMAGE]
                    for j in range(1, self.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
                        all_masks[j][i] = all_masks[j][i][keep, :]

            if vis:

                visualization_path = vis_path if vis_path else os.path.join(self.cfg.TEST.VISUALIZATION_PATH,
                                                                            cache_name)
                if not os.path.isdir(visualization_path):
                    os.makedirs(visualization_path)
                import cv2
                im = cv2.cvtColor(cv2.imread(self.roidb[i]['image']), cv2.COLOR_BGR2RGB)

                visualize_masks(im,
                               [[]] + [all_boxes[j][i] for j in range(1, self.num_classes)],
                               [[]] + [all_masks[j][i] for j in range(1, self.num_classes)],
                               1.0,
                               self.cfg.network.PIXEL_MEANS, self.class_names, threshold=0.5,
                               save_path=os.path.join(visualization_path, '{}{}'.format(vis_name if vis_name else i,
                                                                                        vis_ext)), transform=False)
        if cache_name:
            cache_path = os.path.join(self.result_path, cache_name)
            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)
            cache_path = os.path.join(cache_path, 'detections.pkl')
            self.show_info('Done! Saving detections into: {}'.format(cache_path))
            with open(cache_path, 'wb') as detfile:
                cPickle.dump(all_boxes, detfile)
        return all_boxes, all_masks

    def aggregate(self, scale_cls_dets, scale_cls_masks, vis=False, cache_name='cache', vis_path=None, vis_name=None,
                  pre_nms_db_divide=10, vis_ext='.png'):
        # TODO: finish the multi process version, fix bug
        n_scales = len(scale_cls_dets)
        assert n_scales == len(self.cfg.TEST.VALID_RANGES), 'A valid range should be specified for each test scale'
        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        all_masks = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        nms_pool = Pool(32)
        if len(scale_cls_dets) > 1:
            self.show_info('Aggregating detections from multiple scales and applying NMS...')
        else:
            self.show_info('Performing NMS on detections...')

        # Apply ranges and store detections per category
        parallel_nms_args = [[] for _ in range(pre_nms_db_divide)]
        parallel_nms_mask_args = [[] for _ in range(pre_nms_db_divide)]
        n_roi_per_pool = math.ceil(self.num_images / float(pre_nms_db_divide))

        for i in range(self.num_images):
            for j in range(1, self.num_classes):
                agg_dets = np.empty((0, 5), dtype=np.float32)
                agg_masks = np.empty((0, 28, 28), dtype=np.float32)
                for all_cls_dets, all_cls_masks, valid_range in zip(scale_cls_dets, scale_cls_masks, self.cfg.TEST.VALID_RANGES):
                    cls_dets = all_cls_dets[j][i]
                    cls_masks = all_cls_masks[j][i]
                    heights = cls_dets[:, 2] - cls_dets[:, 0]
                    widths = cls_dets[:, 3] - cls_dets[:, 1]
                    areas = widths * heights
                    lvalid_ids = np.where(areas > valid_range[0] * valid_range[0])[0] if valid_range[0] > 0 else \
                        np.arange(len(areas))
                    uvalid_ids = np.where(areas <= valid_range[1] * valid_range[1])[0] if valid_range[1] > 0 else \
                        np.arange(len(areas))
                    valid_ids = np.intersect1d(lvalid_ids, uvalid_ids)
                    cls_dets = cls_dets[valid_ids, :] if len(valid_ids) > 0 else cls_dets
                    cls_masks = cls_masks[valid_ids, :, :] if len(valid_ids) > 0 else cls_masks
                    agg_dets = np.vstack((agg_dets, cls_dets))
                    agg_masks = np.concatenate((agg_masks, cls_masks), axis=0)
                parallel_nms_args[int(i / n_roi_per_pool)].append(agg_dets)
                parallel_nms_mask_args[int(i / n_roi_per_pool)].append(agg_masks)
        # Divide roidb and perform NMS in parallel to reduce the memory usage
        im_offset = 0
        for part in tqdm(range(pre_nms_db_divide)):
            final_dets = nms_pool.map(self.nms_worker.worker, parallel_nms_args[part])
            n_part_im = int(len(final_dets) / (self.num_classes - 1))
            for i in range(n_part_im):
                for j in range(1, self.num_classes):
                    if self.cfg.TEST.NMS < 0:
                        all_boxes[j][im_offset + i] = final_dets[i * (self.num_classes - 1) + (j - 1)]
                        # TODO: finish the code for masks
                    else:
                        # pdb.set_trace()
                        # TODO: finish the multi process code
                        all_boxes[j][im_offset + i] = parallel_nms_args[part][i * (self.num_classes - 1) + (j - 1)][final_dets[i * (self.num_classes - 1) + (j - 1)], :]
                        all_masks[j][im_offset + i] = parallel_nms_mask_args[part][i * (self.num_classes - 1) + (j - 1)][final_dets[i * (self.num_classes - 1) + (j - 1)], :]
            im_offset += n_part_im

        nms_pool.close()
        # Limit number of detections to MAX_PER_IMAGE if requested and visualize if vis is True
        for i in range(self.num_images):
            if self.cfg.TEST.MAX_PER_IMAGE > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, self.num_classes)])
                if len(image_scores) > self.cfg.TEST.MAX_PER_IMAGE:
                    image_thresh = np.sort(image_scores)[-self.cfg.TEST.MAX_PER_IMAGE]
                    for j in range(1, self.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
                        all_masks[j][i] = all_masks[j][i][keep, :]
            if vis:

                visualization_path = vis_path if vis_path else os.path.join(self.cfg.TEST.VISUALIZATION_PATH,
                                                                            cache_name)
                if not os.path.isdir(visualization_path):
                    os.makedirs(visualization_path)
                import cv2

                im = cv2.cvtColor(cv2.imread(self.roidb[i]['image']), cv2.COLOR_BGR2RGB)
                visualize_masks(im,
                               [[]] + [all_boxes[j][i] for j in range(1, self.num_classes)],
                               [[]] + [all_masks[j][i] for j in range(1, self.num_classes)],
                               1.0,
                               self.cfg.network.PIXEL_MEANS, self.class_names, threshold=0.5,
                               save_path=os.path.join(visualization_path, '{}{}'.format(vis_name if vis_name else i,
                                                                                        vis_ext)), transform=False)

        if cache_name:
            cache_path = os.path.join(self.result_path, cache_name)
            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)
            cache_path = os.path.join(cache_path, 'detections.pkl')
            self.show_info('Done! Saving detections into: {}'.format(cache_path))
            with open(cache_path, 'wb') as detfile:
                cPickle.dump(all_boxes, detfile)
        return all_boxes

    def get_detections(self, cls_thresh=1e-3, cache_name='cache', evaluate=False, vis=False, vis_path=None,
                       vis_ext='.png'):
        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)] # (19,num_images,0) # Probably replace num_images with batch_size?
        all_masks = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)] # (19,num_images,0)
        data_counter = 0
        detect_time, post_time = 0, 0
        if vis:
            visualization_path = vis_path if vis_path else os.path.join(self.cfg.TEST.VISUALIZATION_PATH, cache_name)

        if vis and not os.path.isdir(self.cfg.TEST.VISUALIZATION_PATH):
            os.makedirs(self.cfg.TEST.VISUALIZATION_PATH)

        
        for batch in self.test_iter:
            # pdb.set_trace()
            im_info = batch.data[1].asnumpy()
            scales = im_info[:, 2].reshape(-1, self.batch_size)
            # Run detection on the batch
            stime = time.time()
            scores, boxes, data, im_ids, masks = self.detect(batch, scales)
            detect_time += time.time() - stime

            stime = time.time()
            for i, (cscores, cboxes, im_id, cmasks) in enumerate(zip(scores, boxes, im_ids, masks)):
                parallel_nms_args = []
                masks_list = []
                for j in range(1, self.num_classes):
                    # Apply the score threshold
                    inds = np.where(cscores[:, j] > cls_thresh)[0]
                    rem_scores = cscores[inds, j, np.newaxis]
                    rem_boxes = cboxes[inds, 0:4]
                    # pdb.set_trace()
                    cls_masks = cmasks[inds, 1, :, :]

                    # rem_masks = cmasks[inds, :]
                    cls_dets = np.hstack((rem_boxes, rem_scores))
                    if evaluate or vis:
                        parallel_nms_args.append(cls_dets)
                        masks_list.append(cls_masks)
                    else:
                        all_boxes[j][im_id] = cls_dets
                        all_masks[j][im_id] = cls_masks

                # Apply nms
                if evaluate or vis:
                    if not self.thread_pool:
                        self.thread_pool = ThreadPool(8)
                    if self.cfg.TEST.NMS < 0:
                        # TODO: finish vis code for mask
                        final_dets = self.thread_pool.map(self.nms_worker.worker, parallel_nms_args)
                        pdb.set_trace()
                        for j in range(1, self.num_classes):
                            all_boxes[j][im_id] = final_dets[j - 1]
                    else:
                        keeps = self.thread_pool.map(self.nms_worker.worker, parallel_nms_args)
                        for j in range(1, self.num_classes):
                            all_boxes[j][im_id] = all_boxes[j][im_id][keeps[j - 1], :]
                            all_masks[j][im_id] = all_masks[j][im_id][keeps[j - 1], :]
                # Filter boxes based on max_per_image if needed
                if evaluate and self.cfg.TEST.MAX_PER_IMAGE:
                    print(im_id)
                    image_scores = np.hstack([all_boxes[j][im_id][:, -1]
                                              for j in range(1, self.num_classes)])
                    if len(image_scores) > self.cfg.TEST.MAX_PER_IMAGE:
                        image_thresh = np.sort(image_scores)[-self.cfg.TEST.MAX_PER_IMAGE]
                        for j in range(1, self.num_classes):
                            keep = np.where(all_boxes[j][im_id][:, -1] >= image_thresh)[0]
                            all_boxes[j][im_id] = all_boxes[j][im_id][keep, :]
                            all_masks[j][im_id] = all_masks[j][im_id][keep, :]
                if vis:
                    # TODO: finish vis code
                    if not os.path.isdir(visualization_path):
                        os.makedirs(visualization_path)
                    # visualize_dets(batch.data[0][i].asnumpy(),
                    #                [[]] + [all_boxes[j][im_id] for j in range(1, self.num_classes)], im_info[i, 2],
                    #                self.cfg.network.PIXEL_MEANS, self.class_names, threshold=0.5,
                    #                save_path=os.path.join(visualization_path, '{}{}'.format(im_id, vis_ext)))
#                     cv2.imwrite(os.path.join(visualization_path, '{}{}'.format(im_id, vis_ext)), batch.data[0][i].asnumpy())
                    
                    visualize_masks(batch.data[0][i].asnumpy(),
                                   [[]] + [all_boxes[j][im_id] for j in range(1, self.num_classes)],
                                    [[]] + [all_masks[j][im_id] for j in range(1, self.num_classes)],
                                    im_info[i, 2],
                                   self.cfg.network.PIXEL_MEANS, self.class_names, threshold=0.5,
                                   save_path=os.path.join(visualization_path, '{}{}'.format(im_id, vis_ext)))

            data_counter += self.test_iter.get_batch_size()
            post_time += time.time() - stime
            if self.verbose:
                self.show_info('Tester: {}/{}, Detection: {:.4f}s, Post Processing: {:.4}s'.format(
                    min(data_counter, self.num_images),
                    self.num_images, detect_time / data_counter,
                    post_time / data_counter))
        if self.thread_pool:
            self.thread_pool.close()

        return all_boxes, all_masks

    def extract_proposals(self, n_proposals=300, cache_name='cache', vis=False, vis_ext='.png'):
        all_boxes = [[] for _ in range(self.num_images)]
        data_counter = 0
        detect_time, post_time = 0, 0
        if vis and not os.path.isdir(self.cfg.TEST.VISUALIZATION_PATH):
            os.makedirs(self.cfg.TEST.VISUALIZATION_PATH)

        for batch in self.test_iter:
            im_info = batch.data[1].asnumpy()
            scales = im_info[:, 2].reshape(-1, self.batch_size)
            # Run detection on the batch
            stime = time.time()
            scores, boxes, data, im_ids = self.get_proposals(batch, scales)
            detect_time += time.time() - stime

            stime = time.time()
            for i, (cscores, cboxes, im_id) in enumerate(zip(scores, boxes, im_ids)):
                # Keep the requested number of rois
                rem_scores = cscores[0:n_proposals, np.newaxis]
                rem_boxes = cboxes[0:n_proposals, 0:4]
                cls_dets = np.hstack((rem_boxes, rem_scores)).astype(np.float32)
                if vis:
                    visualization_path = os.path.join(self.cfg.TEST.VISUALIZATION_PATH, cache_name)
                    if not os.path.isdir(visualization_path):
                        os.makedirs(visualization_path)
                    visualize_dets(batch.data[0][i].asnumpy(),
                                   [[]] + [cls_dets], im_info[i, 2],
                                   self.cfg.network.PIXEL_MEANS, ['__background__', 'object'], threshold=0.5,
                                   save_path=os.path.join(visualization_path, '{}{}'.format(im_id, vis_ext)))
                all_boxes[im_id] = cls_dets
            data_counter += self.test_iter.get_batch_size()
            post_time += time.time() - stime
            self.show_info(
                'Tester: {}/{}, Forward: {:.4f}s, Post Processing: {:.4}s'.format(min(data_counter, self.num_images),
                                                                                  self.num_images,
                                                                                  detect_time / data_counter,
                                                                                  post_time / data_counter))
        return all_boxes


def detect_scale_worker(arguments):
    [scale, nbatch, context, config, sym_def, \
     roidb, imdb, arg_params, aux_params, vis] = arguments
    print('Performing inference for scale: {}'.format(scale))
    nGPUs = len(context)
    sym_inst = sym_def(n_proposals=400, test_nbatch=nbatch)
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=nGPUs * nbatch, nGPUs=nGPUs, threads=32,
                               pad_rois_to=400, crop_size=None, test_scale=scale,num_classes=imdb.num_classes)
    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in test_iter.provide_data_single],
                        label_names=None)
    mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)
    # Create Tester
    tester = Tester(mod, imdb, roidb, test_iter, cfg=config, batch_size=nbatch)
    return tester.get_detections(vis=(vis and config.TEST.VISUALIZE_INTERMEDIATE_SCALES),
                                 evaluate=False, cache_name='dets_scale_{}x{}'.format(scale[0], scale[1]))


def imdb_detection_wrapper(sym_def, config, imdb, roidb, context, arg_params, aux_params, vis):
    if vis and config.TEST.CONCURRENT_JOBS > 1:
        print('Visualization is only allowed with 1 CONCURRENT_JOBS')
        print('Setting CONCURRENT_JOBS to 1')
        config.TEST.CONCURRENT_JOBS = 1
    detections = []
    masks = []
    if config.TEST.CONCURRENT_JOBS == 1:
        for nbatch, scale in zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES):
            detection, mask = detect_scale_worker([scale, nbatch, context, config, sym_def, \
                                                   roidb, imdb, arg_params, aux_params, vis])
            detections.append(detection)
            masks.append(mask)
    else:
        im_per_job = int(math.ceil(float(len(roidb)) / config.TEST.CONCURRENT_JOBS))
        roidbs = []
        pool = Pool(config.TEST.CONCURRENT_JOBS)
        for i in range(config.TEST.CONCURRENT_JOBS):
            roidbs.append([roidb[j] for j in range(im_per_job * i, min(im_per_job * (i + 1), len(roidb)))])

        for _, (nbatch, scale) in enumerate(zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES)):
            parallel_args = []
            for j in range(config.TEST.CONCURRENT_JOBS):
                parallel_args.append([scale, nbatch, context, config, sym_def, \
                                      roidbs[j], imdb, arg_params, aux_params, vis])

            detection_list = pool.map(detect_scale_worker, parallel_args)
            tmp_dets = detection_list[0]
            for i in range(1, len(detection_list)):
                for j in range(imdb.num_classes):
                    tmp_dets[j] += detection_list[i][j]

            # Cache detections...
            cache_path = os.path.join(imdb.result_path, 'dets_scale_{}x{}'.format(scale[0], scale[1]))
            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)
            cache_path = os.path.join(cache_path, 'detections.pkl')
            print('Done! Saving detections into: {}'.format(cache_path))
            with open(cache_path, 'wb') as detfile:
                cPickle.dump(tmp_dets, detfile)
            detections.append(tmp_dets)
        pool.close()

    tester = Tester(None, imdb, roidb, None, cfg=config, batch_size=nbatch)
    all_boxes, all_masks = tester.aggregateSingle(detections, masks, vis=vis, vis_path="./data/demo_batch/batch_results",cache_name='dets_final')

    print('Evaluating detections...')
    imdb.evaluate_detections(all_boxes, 'segm', all_masks)
    print('All done!')


def proposal_scale_worker(arguments):
    [scale, nbatch, context, config, sym_def, \
     roidb, imdb, arg_params, aux_params, vis] = arguments
    print('Performing inference for scale: {}'.format(scale))
    nGPUs = len(context)
    sym_inst = sym_def(n_proposals=400, test_nbatch=nbatch)
    sym = sym_inst.get_symbol_rpn(config, is_train=False)
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=nGPUs * nbatch, nGPUs=nGPUs, threads=32,
                               pad_rois_to=400, crop_size=None, test_scale=scale,num_classes=imdb.num_classes)
    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in test_iter.provide_data_single],
                        label_names=None)
    mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)
    # Create Tester
    tester = Tester(mod, imdb, roidb, test_iter, cfg=config, batch_size=nbatch)
    return tester.extract_proposals(vis=(vis and config.TEST.VISUALIZE_INTERMEDIATE_SCALES),
                                    cache_name='props_scale_{}x{}'.format(scale[0], scale[1]))


def imdb_proposal_extraction_wrapper(sym_def, config, imdb, roidb, context, arg_params, aux_params, vis):
    if vis and config.TEST.CONCURRENT_JOBS > 1:
        print('Visualization is only allowed with 1 CONCURRENT_JOBS')
        print('Setting CONCURRENT_JOBS to 1')
        config.TEST.CONCURRENT_JOBS = 1

    proposals = []
    if config.TEST.CONCURRENT_JOBS == 1:
        for nbatch, scale in zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES):
            proposals.append(proposal_scale_worker([scale, nbatch, context, config, sym_def, \
                                                    roidb, imdb, arg_params, aux_params, vis]))
    else:
        im_per_job = int(math.ceil(float(len(roidb)) / config.TEST.CONCURRENT_JOBS))
        roidbs = []
        pool = Pool(config.TEST.CONCURRENT_JOBS)
        for i in range(config.TEST.CONCURRENT_JOBS):
            roidbs.append([roidb[j] for j in range(im_per_job * i, min(im_per_job * (i + 1), len(roidb)))])

        for i, (nbatch, scale) in enumerate(zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES)):
            parallel_args = []
            for j in range(config.TEST.CONCURRENT_JOBS):
                parallel_args.append([scale, nbatch, context, config, sym_def, \
                                      roidbs[j], imdb, arg_params, aux_params, vis])

            proposal_list = pool.map(proposal_scale_worker, parallel_args)
            tmp_props = []
            for prop in proposal_list:
                tmp_props += prop

            # Cache proposals...
            cache_path = os.path.join(imdb.result_path, 'props_scale_{}x{}'.format(scale[0], scale[1]))
            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)
            cache_path = os.path.join(cache_path, 'proposals.pkl')
            print('Done! Saving proposals into: {}'.format(cache_path))
            with open(cache_path, 'wb') as detfile:
                cPickle.dump(tmp_props, detfile)

            proposals.append(tmp_props)
        pool.close()

    if not os.path.isdir(config.TEST.PROPOSAL_SAVE_PATH):
        os.makedirs(config.TEST.PROPOSAL_SAVE_PATH)

    final_proposals = proposals[0]

    if len(proposals) > 1:
        for i in range(len(proposals[0])):
            for j in range(1, len(proposals)):
                final_proposals[i] = np.vstack((final_proposals[i], proposals[j][i]))
    save_path = os.path.join(config.TEST.PROPOSAL_SAVE_PATH, '{}_{}_rpn.pkl'.format(config.dataset.dataset.upper(), config.dataset.test_image_set))
    with open(save_path, 'wb') as file:
        cPickle.dump(final_proposals, file)

    ('All done!')
