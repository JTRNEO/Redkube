import sys
#sys.path.insert(0, './sniper/SNIPER-mxnet')
import mxnet as mx
import argparse
import sys
sys.path.insert(0, '/sniper/service/models/SNIPER/sniper_utils/lib')
sys.path.insert(0, '/sniper/service/models/SNIPER/sniper_utils/symbols')
sys.path.insert(0, '/sniper/service/models/SNIPER/sniper_utils/configs')
sys.path.insert(0, '/sniper/service/models/SNIPER/sniper_utils')
sys.path.insert(0, './')
import os
#os.chdir("..")
import pdb
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference_mask import Tester
from skimage.measure import find_contours, approximate_polygon
import cv2
import math
import numpy as np
from symbols.faster import *
from configs.faster.default_configs import update_config, update_config_from_list
from configs.faster.default_configs import config
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='./configs/faster/sniper_res101_e2e_mask_pred_satellite.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()


class SNIPER():
    def __init__(self, modelpath):
        self.model_path = modelpath
        

        #self.args = parser()

        self.cfg = '/sniper/service/models/SNIPER/sniper_utils/configs/faster/sniper_res101_e2e_mask_pred_satellite.yml'
        self.save_prefix = "SNIPER"
        

        update_config(self.cfg)

        self.satellite_names = [u'BG', u'Planes', u'Ships', u'Helicopter', u'Vehicles', u'Bridges', u'Buildings',
                       u'Parking Lots', u'Satellite Dish', u'Solar Panels', u'Storage Tank', u'Swimming Pool',
                       u'Sports Stadium/Field', u'Shipping Containers', u'Crane', u'Train', u'Mil Vehicles',
                       u'Missiles/Missile Systems', u'Comms Towers']
        # self.satellite_names = [u'BG', u'Planes', u'Ships', u'Helicopter', u'Vehicles', u'Buildings',
        #                u'Parking Lots',  u'Storage Tank', u'Swimming Pool',
        #                u'Sports Stadium/Field', u'Shipping Containers', u'Crane',  u'Comms Towers']

        assert config.dataset.NUM_CLASSES == len(self.satellite_names), "Incorrect specification of classes"


        # Use just the first GPU for demo
        self.context = [mx.gpu(int(config.gpus[0]))]
        config.output_path = "/sniper/service/checkpoint/sniper_res_101_bn_mask_satellite_18"
        if not os.path.isdir(config.output_path):
            os.mkdir(config.output_path)

        logger, output_path = create_logger(config.output_path, self.cfg, config.dataset.image_set)
        # Pack db info
        self.db_info = EasyDict()
        self.db_info.name = 'coco'
        self.db_info.result_path = 'data/demo'

        self.db_info.classes = self.satellite_names
        self.db_info.num_classes = len(self.db_info.classes)

        # Create the model
        sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
        self.sym_inst = sym_def(n_proposals=400, test_nbatch=1)
        self.sym = self.sym_inst.get_symbol_rcnn(config, is_train=False)
        self.model_prefix = os.path.join(output_path, self.save_prefix)
        config.TEST.TEST_EPOCH = 30
        self.arg_params, self.aux_params = load_param(self.model_prefix, config.TEST.TEST_EPOCH,
                                            convert=True, process=True)
        
        print("Loading {}_{}.params".format(self.model_prefix, config.TEST.TEST_EPOCH))

    def forward(self, image, imageId):

        # Get image dimensions
        # TODO: fix the tmp save later
        #tmp_save_path = './data/demo/' + imageId + '.png'

        #cv2.imwrite(tmp_save_path, image)
        #width, height = Image.open(tmp_save_path).size

        width, height = image.shape[0],image.shape[1]


        # Pack image info
        roidb = [{'image': image, 'width': width, 'height': height, 'flipped': False}]

        # Creating the Logger
        test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                                   crop_size=None, test_scale=config.TEST.SCALES[0],
                                   num_classes=self.db_info.num_classes)
        # Create the module
        shape_dict = dict(test_iter.provide_data_single)
        self.sym_inst.infer_shape(shape_dict)
        mod = mx.mod.Module(symbol=self.sym,
                            context=self.context,
                            data_names=[k[0] for k in test_iter.provide_data_single],
                            label_names=None)
        mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)

        # Initialize the weights

        mod.init_params(arg_params=self.arg_params, aux_params=self.aux_params)

        # Create the tester
        tester = Tester(mod, self.db_info, roidb, test_iter, cfg=config, batch_size=1)

        # Sequentially do detection over scales
        # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
        all_detections = []
        all_masks = []
        for s in config.TEST.SCALES:
            # Set tester scale
            tester.set_scale(s)
            # Perform detection
            detections, masks = tester.get_detections(vis=False, evaluate=False, cache_name=None)
            all_detections.append(detections)
            all_masks.append(masks)
            # all_detections.append(tester.get_detections(vis=False, evaluate=False, cache_name=None))

        # Aggregate results from multiple scales and perform NMS
        tester = Tester(None, self.db_info, roidb, None, cfg=config, batch_size=1)
        #file_name, out_extension = "blah","png"#os.path.splitext(os.path.basename(tmp_save_path))
        # all_detections, all_masks = tester.aggregateSingle(all_detections, all_masks, vis=False, cache_name=None,
        #                                                    vis_path='./data/demo/',
        #                                                    vis_name='{}_detections'.format(file_name),
        #                                                    vis_ext=out_extension)
        all_detections, all_masks = tester.aggregateSingleService(all_detections, all_masks, vis=False, cache_name=None)

        return all_detections, all_masks



    def infere(self, image, imageId=None, debug=False, pixel_size=(0.3,0.3)):
            import timeit
            #tic=timeit.default_timer()

            data = self.forward(image, imageId)
            #toc=timeit.default_timer()
            #print("Time for forward pass:",toc - tic)

            all_detections, all_masks = data
            # pdb.set_trace()
            result = []

            tic1=timeit.default_timer()

            #print("DEtection DIM=",np.array(all_detections).shape)
            #print("DET 1=",np.array(all_detections[0]).shape)
            # print("LEN =",len(all_detections))
            # print("DET=",len(all_detections[1][0]))
            # print(all_detections)
            # # import pickle
            # # with open('det.pkl','wb') as f:
            # #     print("Dumping detections")
            # #     pickle.dump(all_detections,f)
            # print(type(all_detections[1][0]))
            # print("DET SHAPE = ",all_detections[1][0].shape)
           
            # print("LEN =",len(all_masks))
            # print("MASK=",len(all_masks[1][0]))
            # print(all_masks[1][0])
            # print(type(all_masks[1][0]))
            # print("MASK SHAPE = ",all_masks[1][0].shape)
            
            # boxes, segms, classes = self.convert_from_cls_format(all_detections, all_masks)

            # print("BOX=",boxes.shape)
            # print("SEGMS=",segms.shape)
            # print("CLASS=",classes.shape)
            
            for j in range(len(all_detections)):
                if j == 0:
                    # pass bg class
                    pass

                cls_dets = all_detections[j][0]
                cls_masks = all_masks[j][0]
                class_id = j
                label = self.satellite_names[class_id]
                tic2=timeit.default_timer()
                for i in range(len(cls_dets)):
                    score = cls_dets[i, 4]
                    if score < 0.7:
                        continue
                    bbox = cls_dets[i, :4]
                    mask_image = np.zeros((1024, 1024))
                    mask = cls_masks[i, :, :]
                    # paste mask
                    bbox = map(int, bbox)
                    bbox[0] = max(bbox[0], 1)
                    bbox[1] = max(bbox[1], 1)
                    bbox[2] = min(bbox[2], 1024 - 1)
                    bbox[3] = min(bbox[3], 1024 - 1)
                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    # pdb.set_trace()
                    tic3 = timeit.default_timer()
                    area, perimetr, cv2Poly = self.getMaskInfo(mask_image, (10, 10), pixel_size=pixel_size)
                    toc3 = timeit.default_timer()
                    #print("Time for getting Mask Info=",toc3 - tic3)

                    if cv2Poly is None:
                        # print("Warning: Object is recognized, but contour is empty!")
                        continue

                    verts = cv2Poly[:, 0, :]
                    r = {'classId': class_id,
                         'score': score,
                         'label': label,
                         'area': area,
                         'perimetr': perimetr,
                         'verts': verts}

                    if imageId is not None:
                        r['objId'] = "{}_obj-{}".format(imageId, i)

                    result.append(r)
                toc2=timeit.default_timer()
                #print("Time for inner loop=",toc2 - tic2)

            toc1=timeit.default_timer()

            #print("Time for inference loop:",toc1 - tic1)

            return result

    def getMaskInfo(self, img, kernel=(10, 10), pixel_size=(0.3,0.3)):

        #Define kernel
        kernel = np.ones(kernel, np.uint8)

        #Open to erode small patches
        #thresh = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #Close little holes
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=4)

        #thresh=thresh.astype('uint8')
        # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(img.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        maxArea = 0
        maxContour = None

        # Get largest area contour
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > maxArea:
                maxArea = a
                maxContour = cnt

        if maxContour is None: return [None, None, None]

        perimeter = cv2.arcLength(maxContour,True)

        # TODO: Return this
        # Get area in meters squared
        pix_sq = pixel_size[0] * pixel_size[1]
        maxArea_m2 = maxArea * pix_sq 
        maxArea_m2 = round(maxArea_m2, 3)

        # aproximate contour with the 1% of squared perimiter accuracy
        # approx = cv2.approxPolyDP(maxContour, 0.01*math.sqrt(perimeter), True)

        return maxArea, perimeter, maxContour





