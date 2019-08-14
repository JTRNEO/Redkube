# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference Module
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from symbols.faster import *
from configs.faster.default_configs import config, update_config, update_config_from_list
from data_utils.load_data import load_proposal_roidb
import os
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
from inference_mask import imdb_detection_wrapper
from inference_mask import imdb_proposal_extraction_wrapper
import os
import time
from PIL import Image
from easydict import EasyDict

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER test module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    			    default='/sniper/service/models/SNIPER/sniper_utils/configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--img_dir_path', dest='img_dir_path', help='Path to the image directory', type=str,
                            default='data/demo_batch/images')
    arg_parser.add_argument('--dataset', dest='dataset', help='choose categories belong to which dataset', type=str,
                            default='satellite')
    arg_parser.add_argument('--vis', dest='vis', help='Whether to visualize the detections',
                            action='store_true')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()


def get_config():
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    import yaml
    from easydict import EasyDict as edict
    with open(args.cfg) as f:
        exp_config = edict(yaml.load(f))
    return exp_config


def main():
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

   
    # Create roidb
    roidb, imdb = load_proposal_roidb(config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path,
                                     config.dataset.dataset_path,
                                     proposal=config.dataset.proposal, only_gt=True, flip=False,
                                     result_path=config.output_path,
                                     proposal_path=config.proposal_path, get_imdb=True)

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    model_prefix = os.path.join(output_path, args.save_prefix)
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                        convert=True, process=True)

    sym_inst = eval('{}.{}'.format(config.symbol, config.symbol))
    if config.TEST.EXTRACT_PROPOSALS:
        imdb_proposal_extraction_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)
    else:
        imdb_detection_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)

if __name__ == '__main__':
    main()


