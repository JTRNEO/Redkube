#export LD_LIBRARY_PATH=/home/gxdai/nvidia/cuda-9.0/lib64:/home/gxdai/nvidia/cudnn74/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/gxdai/nvidia/cuda-9.0/lib64/stubs:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=7 python main_test_mask.py --cfg configs/faster/sniper_res101_e2e_mask_pred_satellite.yml --img_dir_path data/Satellite/images/val --dataset Satellite --set TEST.TEST_EPOCH 10

