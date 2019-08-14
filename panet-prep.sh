#!/bin/bash
pod=$(kubectl get pod | grep satellite-preprocessing | awk -F " " '{print $1}')
podprefix="kubectl exec -it "
podcom=" -- "
commander=${podprefix}${pod}${podcom}
alias kes="$commander"

podcount=10


#kes rm -rf /mnt/geojson
#kes mkdir /mnt/geojson

# sh blah.sh -path(p) -m(model)
# while getopts ":p:" opt 
# do 
#  case $opt in
#    p)
#    impath=${OPTARG}
#    datapath=${OPTARG}
#    datapath=${datapath%/*}  #os dirname
#    datapath=${datapath##*/} #os basename
#    prefix="/mnt/"
#    befix="/"
#    ;;
#  esac
# done

# /home/ashwin/Desktop/New_Data/Davis_Monthan/Davis_Monthan_AFB_20180814.tif
# ${datapath%/*} - /home/ashwin/Desktop/New_Data/Davis_Monthan/
# ${datapath##*/} - Davis_Monthan_AFB_20180814.tif

# # Clear main and processing queues
# kes redis-cli -h redis del job:processing exit
# kes redis-cli -h redis del job exit

# kes redis-cli -h redis del result:processing exit
# kes redis-cli -h redis del result exit

# kes redis-cli -h redis del status:processing exit
# kes redis-cli -h redis del status exit

# # Crop
# kes rm -rf /mnt/crops
# echo 'begin crop'
# kes python /mnt/scripts/crop_service.py --dataset /mnt/sat_images/Davis_Monthan_AFB_20180814.tif --image_dir /mnt/sat_images2 --image_size 1024 --yaml /mnt/yamls/jobservice_panet.yaml
# echo 'finish crop'
# # Inference
# echo 'Begin inference'
# kubectl create -f /home/ubuntu/redkube/models/panet/model_yamls/jobservice_panet.yaml
# # judge_queue checks the status of the queue and if it's empty, waits for 30 seconds
# kes python /mnt/scripts/judge_queue.py --queue job 
# kes python /mnt/scripts/judge_queue.py --queue result
# kubectl delete job panet-inference
# echo 'Finished!'

DIR='/home/ubuntu/redkube/nfs/sat_images'

for IMAGE in $DIR/*.tif* 
do
	# Puts impath in the nfs fodler so its accessbile by the container
	# $IMAGE -> gets you the fullpath
	# ${IMAGE##*/} -> gets you the dirname 
	# ${IMAGE##*/} -> gets you the filename 
	echo /mnt/sat_images/${IMAGE##*/} | tee /home/ubuntu/redkube/nfs/impath.txt

	echo 'Clearing queues'

	# Clear main and processing queues
	kes redis-cli -h redis del job:processing exit
	kes redis-cli -h redis del job exit

	kes redis-cli -h redis del result:processing exit
	kes redis-cli -h redis del result exit

	kes redis-cli -h redis del status:processing exit
	kes redis-cli -h redis del status exit

	# Crop
	kes rm -rf /mnt/crops
	echo 'begin crop'
	kes python /mnt/scripts/crop_service.py --dataset /mnt/sat_images/${IMAGE##*/} --image_dir /mnt/sat_images/ --image_size 1024 --pod $podcount
	# Create a path file to point to the image location
	echo 'finish crop'

	echo 'Setting up pod specifications'
	python /home/ubuntu/redkube/nfs/scripts/pod_setup.py --pods $podcount --gpu 1 --yaml /home/ubuntu/redkube/models/panet/model_yamls/jobservice_panet.yaml
	echo 'Pod Setup Complete'

	# Inference
	echo 'Begin inference'
	kubectl create -f /home/ubuntu/redkube/models/panet/model_yamls/jobservice_panet.yaml
	# judge_queue checks the status of the queue and if it's empty, waits for 30 seconds
	kes python /mnt/scripts/judge_queue.py --queue job 
	kes python /mnt/scripts/judge_queue.py --queue result
	kubectl delete job panet-inference
	echo 'Finished!'

done


 
   





