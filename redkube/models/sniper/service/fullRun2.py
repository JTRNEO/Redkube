from tqdm import tqdm
import copy
import pickle
import os
import importlib
import math
from pydoc import locate

import sys
sys.path.insert(0,'/mnt/scripts')

import rediswq
import redis
import cv2


import re
import time
import os
import logging
#os.chdir("/sniper/service")
#sys.path.append('/sniper/SNIPER-mxnet/')

#from GeoImageKube import GeoImage
from GeoImage import GeoImage
from ContourProcessor import ContourProcessor
from config import serviceConfig, Cache

import pdb
logging.basicConfig(level=logging.DEBUG)


def run_process(serviceConfig, outputDir, Model):

    logging.debug('IMAGE:{}'.format(os.path.basename(os.path.normpath(serviceConfig.image_path))))

    if os.path.isdir(outputDir) is False:
        os.makedirs(outputDir)
    
    cache_path = os.path.join(outputDir,'full_results.pkl')

    geoImg = GeoImage(serviceConfig.image_path, gap=200)
    
    px, py = geoImg.pixelSize # tuple(px,py)
    pixel_size = (px,py)
    
    model = None if serviceConfig.cacheStrategy == Cache.read else Model(serviceConfig.modelStorePath)

    geojson = ContourProcessor(geoImg, outputDir)

    readCacheIterator = None
    if serviceConfig.cacheStrategy == Cache.read:
        with open(cache_path, 'rb') as input:
            cachedReadResults = pickle.load(input)
        readCacheIterator = iter(cachedReadResults)


    cacheWriteResults = []

    #pdb.set_trace()

    slice_ids = geoImg.getSplits()

    # Cache Read mode

    if serviceConfig.cacheStrategy == Cache.read:
        for xyOffset in slice_ids:
            result = next(readCacheIterator)

    # Cache Write mode - Use multi-gpu
    if serviceConfig.cacheStrategy == Cache.write:

        # k8s redis
        # Redis Queues
        q = rediswq.RedisWQ(name="job", host="redis")
        qresult = rediswq.RedisWQ(name="result", host="redis")
        qstat = rediswq.RedisWQ(name="status", host="redis")

        # Redis Object
        qr = redis.Redis(host='redis', port=6379, decode_responses=True)

        print("Worker with sessionID: " +  q.sessionID())
        print("Initial queue state: empty=" + str(q.empty()))
        start_time=time.time()
        while not q.empty():
            # For ... to not occur and mess up the array to string conversion
            import numpy as np
            np.set_printoptions(threshold=np.inf)

            item = q.lease(lease_secs=60, block=False)
            
            if item is not None:
                    itemstr = item.decode("utf=8")
                    itemstr = re.findall(r"\d+\.?\d*",itemstr)
                    slice_name = "{}_{}.png".format(itemstr[0],itemstr[1])
                    print("Working on slice :" + slice_name )
                    im_path='/mnt/crops/'+ slice_name


                    left, up = itemstr

                    img = cv2.imread(im_path) # Read image from crop directory -> left_up.png
                    result = model.infere(img, imageId='left-{}_up-{}'.format(left, up))

                    
                    # # Default -> Creates race condition
                    # patchGeom = geojson.addPatchBoundary(left, up)
                    # for feature in result:
                    #      geojson.addFeature(left, up, feature)
                    #      #pod_flag = 1
                    # q.complete(item)

                    # Create queue element
                    q_elem = [left, up, result]

                    # Push into receiver queue
                    qr.rpush('result',str(q_elem))
                    q.complete(item)

                   
            else:
                print("Waiting...")       
        
        end_time=time.time()
    
        print("Queue Empty \nExiting","\nProcess Time:",'{}s'.format(end_time-start_time))
        
        if qstat.empty() is False:
            ind = qstat.lease(lease_secs=60, block=False)
            qstat.complete(ind)
        else:
            from numpy import array, int32
            while not qresult.empty():
                res = qresult.lease(lease_secs=60, block=False)
                if res is not None:
                    
                    qres_elem = eval(res)

                    left, up = eval(qres_elem[0]), eval(qres_elem[1])
                    result = qres_elem[2] # type list

                    cacheWriteResults.append(copy.deepcopy(result))

                    patchGeom = geojson.addPatchBoundary(left, up)
                    for feature in result:
                        geojson.addFeature(left, up, feature, patchGeom)

                    qresult.complete(res)

            logging.debug("Caching Detections: {}".format(len(cacheWriteResults)))
            
            with open(cache_path, 'wb') as output:
                pickle.dump(cacheWriteResults, output, pickle.HIGHEST_PROTOCOL)

            logging.debug('Geojson Cleanup')
            geojson.cleanUp()


def main():
    
    # Import your model
    #################################################
    """
    A general way of importing the model, so instead of this:
    >> from service.MaskRCNN import MaskRCNN
    >> from service.PANet import PANet 

    You can just specify the model name in config.py and it's
    imported automatically using the model name.

    """
    from models.SNIPER.Sniper_v2 import SNIPER 
    Model = SNIPER
    MODEL_NAME = "SNIPER"
    #if serviceConfig.cacheStrategy == Cache.write:
    #    MODEL_NAME = serviceConfig.modelName
    #    modulename = "models." + MODEL_NAME + "." + MODEL_NAME
    #    # mod = __import__(modulename, fromlist=[MODEL_NAME])
    #    mod = locate(modulename)
    #    Model = getattr(mod, MODEL_NAME)
    #    #################################################

    print("MODEL:",MODEL_NAME)
    #else:
    #    Model = None

    with open('/mnt/impath.txt') as f:
            serviceConfig.image_path = f.read()[:-1]

    img_name = os.path.basename(os.path.normpath(serviceConfig.image_path))
    print("IMAGE:",img_name)
    outputDir = os.path.join(os.path.dirname(serviceConfig.image_path),'{}/{}/{}'.format(serviceConfig.modelName, serviceConfig.modelVersion, img_name[:-4]))
    run_process(serviceConfig,outputDir,Model)
    



if __name__ == "__main__":
    main()
