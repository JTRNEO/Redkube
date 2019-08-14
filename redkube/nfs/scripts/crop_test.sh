#!/bin/bash
rm -r /mnt/crops
python crop_service.py --dataset /mnt/sat_images/DG_Satellite_AZ_Airfield_20180818.tif --image_dir /sat_images2/ --image_size 1024 --yaml /workspace/PANet/nfs/yamls/jobservice_panet.yaml
