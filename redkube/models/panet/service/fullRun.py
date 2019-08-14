from tqdm import tqdm
import copy
import pickle
import os
import sys
import importlib
from pydoc import locate

sys.path.insert(0,'/mnt/scripts')

import rediswq
import redis
import cv2
import sys

import re
import time
import logging

from GeoImageKube import GeoImage
from ContourProcessor import ContourProcessor
from config import serviceConfig, Cache

sys.path.insert(0, os.getcwd())
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

            # Remove array suppression (...) so that it doesn't mess up the array to string conversion
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
            logging.debug('Writing detections to geojson')
            from numpy import array, int32 # For eval to work
            while not qresult.empty():
                res = qresult.lease(lease_secs=60, block=False)
                if res is not None:

                    res2 = res

                    if sys.version_info[0] == 3:
                        # Python3
                        res2 = res.decode("utf=8")

                    qres_elem = eval(res2)

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

            
            if os.path.exists(geojson.cleanedGeojsonPath) is False:
                logging.error('Cleaned Geojson not produced')




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
    from models.PANet.PANet import PANet
    Model = PANet
    MODEL_NAME = "PANet"
    #MODEL_NAME = serviceConfig.modelName
    #modulename = "service." + "models." + MODEL_NAME + "." + MODEL_NAME 
    ## mod = __import__(modulename, fromlist=[MODEL_NAME])
    #mod = locate(modulename)
    #Model = getattr(mod, MODEL_NAME)
    #################################################

    print("MODEL:",MODEL_NAME)
    #print("IMAGE:",os.path.basename(os.path.normpath(serviceConfig.image_path)))

    img_ext = [".tif",".TIF",".tiff",".TIFF"]

    with open('/mnt/impath.txt') as f:
        serviceConfig.image_path = f.read()[:-1]

    img_name = os.path.basename(os.path.normpath(serviceConfig.image_path))
    print("IMAGE:",img_name)
    outputDir = os.path.join(os.path.dirname(serviceConfig.image_path),'{}/{}/{}'.format(serviceConfig.modelName, serviceConfig.modelVersion, img_name[:-4]))
    



if __name__ == "__main__":
    main()









# from tqdm import tqdm
# import copy
# import pickle
# import os
# import sys
# import importlib
# from pydoc import locate

# sys.path.insert(0,'/mnt/scripts')

# import rediswq
# import redis
# import cv2
# import sys

# import re
# import time

# from GeoImageKube import GeoImage
# from ContourProcessor import ContourProcessor
# from config import serviceConfig, Cache

# sys.path.insert(0, os.getcwd())
# #print(sys.path())

# # def run_process_old(serviceConfig, outputDir, Model):
# #     #outputDir = "/home/ashwin/Desktop/Tripoli_1"#os.path.join(os.path.dirname(serviceConfig.image_path),'{}/{}/'.format(serviceConfig.modelName, serviceConfig.modelVersion))

# #     os.makedirs(outputDir, exist_ok=True)
# #     cache_path = os.path.join(outputDir,'full_results.pkl')

# #     print(serviceConfig.image_path)
# #     geoImg = GeoImage(serviceConfig.image_path, gap=200)
# #     px, py = geoImg.pixelSize # tuple(px,py)
# #     pixel_size = (px,py)
# #     #pixel_size = (round(px,2),round(py,2))
# #     #print("Pixel Size =",pixel_size)
# #     model = None if serviceConfig.cacheStrategy == Cache.read else Model(serviceConfig.modelStorePath)

# #     geojson = ContourProcessor(geoImg, outputDir)

# #     readCacheIterator = None
# #     if serviceConfig.cacheStrategy == Cache.read:
# #         with open(cache_path, 'rb') as input:
# #             cachedReadResults = pickle.load(input)
# #         readCacheIterator = iter(cachedReadResults)


# #     cacheWriteResults = []

# #     for xyOffset in tqdm(geoImg.getSplits(), desc='Processing {}'.format('Model Cache' if serviceConfig.cacheStrategy == Cache.read else 'Image')):

# #         left, up = xyOffset

# #         if serviceConfig.cacheStrategy == Cache.read:
# #             result = next(readCacheIterator)
# #         else:
# #             img = geoImg.getCv2ImgFromSplit(xyOffset)
# #             result = model.infere(img, imageId='left-{}_up-{}'.format(left, up), pixel_size=pixel_size)
# #             if serviceConfig.cacheStrategy == Cache.write: cacheWriteResults.append(copy.deepcopy(result))

# #         patchGeom = geojson.addPatchBoundary(left, up)
# #         for r in result:
# #             geojson.addFeature(left, up, r, patchGeom)
# #             # print("Feature added")

# #     if serviceConfig.cacheStrategy == Cache.write:
# #         with open(cache_path, 'wb') as output:
# #             pickle.dump(cacheWriteResults, output, pickle.HIGHEST_PROTOCOL)


# #     geojson.cleanUp()


# #     # do comparison with the ground truth if it is given
# #     if serviceConfig.groundTruthGeoJson != None:
# #         geojson.compareGT(serviceConfig.groundTruthGeoJson, serviceConfig.gtMappings, serviceConfig.modelMappings)


# def run_process(serviceConfig, outputDir, Model):

#     if os.path.isdir(outputDir) is False:
#         os.makedirs(outputDir)
    
#     cache_path = os.path.join(outputDir,'full_results.pkl')

#     geoImg = GeoImage(serviceConfig.image_path, gap=200)
    
#     px, py = geoImg.pixelSize # tuple(px,py)
#     pixel_size = (px,py)
    
#     model = None if serviceConfig.cacheStrategy == Cache.read else Model(serviceConfig.modelStorePath)

#     geojson = ContourProcessor(geoImg, outputDir)

#     readCacheIterator = None
#     if serviceConfig.cacheStrategy == Cache.read:
#         with open(cache_path, 'rb') as input:
#             cachedReadResults = pickle.load(input)
#         readCacheIterator = iter(cachedReadResults)


#     cacheWriteResults = []

#     #pdb.set_trace()

#     slice_ids = geoImg.getSplits()

#     # Cache Read mode

#     if serviceConfig.cacheStrategy == Cache.read:
#         for xyOffset in slice_ids:
#             result = next(readCacheIterator)

#     # Cache Write mode - Use multi-gpu
#     if serviceConfig.cacheStrategy == Cache.write:

#         # k8s redis
#         # Redis Queues
#         q = rediswq.RedisWQ(name="job", host="redis")
#         qresult = rediswq.RedisWQ(name="result", host="redis")
#         qstat = rediswq.RedisWQ(name="status", host="redis")

#         # Redis Object
#         qr = redis.Redis(host='redis', port=6379, decode_responses=True)

#         print("Worker with sessionID: " +  q.sessionID())
#         print("Initial queue state: empty=" + str(q.empty()))
#         start_time=time.time()
#         while not q.empty():
#             item = q.lease(lease_secs=60, block=False)
            
#             if item is not None:
#                     itemstr = item.decode("utf=8")
#                     itemstr = re.findall(r"\d+\.?\d*",itemstr)
#                     slice_name = "{}_{}.png".format(itemstr[0],itemstr[1])
#                     print("Working on slice :" + slice_name )
#                     im_path='/mnt/crops/'+ slice_name


#                     left, up = itemstr

#                     img = cv2.imread(im_path) # Read image from crop directory -> left_up.png
#                     result = model.infere(img, imageId='left-{}_up-{}'.format(left, up))

                    
#                     # # Default -> Creates race condition
#                     # patchGeom = geojson.addPatchBoundary(left, up)
#                     # for feature in result:
#                     #      geojson.addFeature(left, up, feature)
#                     #      #pod_flag = 1
#                     # q.complete(item)

#                     # Create queue element
#                     q_elem = [left, up, result]

#                     # Push into receiver queue
#                     qr.rpush('result',str(q_elem))
#                     q.complete(item)

                   
#             else:
#                 print("Waiting...")       
        
#         end_time=time.time()
    
#         print("Queue Empty \nExiting","\nProcess Time:",'{}s'.format(end_time-start_time))
        
#         if qstat.empty() is False:
#             ind = qstat.lease(lease_secs=60, block=False)
#             qstat.complete(ind)
#         else:
#             from numpy import array, int32
#             while not qresult.empty():
#                 res = qresult.lease(lease_secs=60, block=False)
#                 if res is not None:

#                     res2 = res

#                     if sys.version_info[0] == 3:
#                         # Python3
#                         res2 = res.decode("utf=8")


#                     # Remove ... which occurs in the results for some reason
#                     res2 = re.sub('\.\.+,\\n\s+','',res2)
                    

#                     # import json
#                     # with open('testb.json','w') as t:
#                     #   json.dump(res,t)

#                     # print(res)
#                     # print(type(res))
#                     # print(len(res))
#                     # print(res[0])

#                     qres_elem = eval(res2)

#                     left, up = eval(qres_elem[0]), eval(qres_elem[1])
#                     result = qres_elem[2] # type list

#                     cacheWriteResults.append(copy.deepcopy(result))

#                     patchGeom = geojson.addPatchBoundary(left, up)
#                     for feature in result:
#                         geojson.addFeature(left, up, feature, patchGeom)

#                     qresult.complete(res)

#             geojson.cleanUp()
            

#         if serviceConfig.cacheStrategy == Cache.write:
#             with open(cache_path, 'wb') as output:
#                 print("Caching Detections")
#                 pickle.dump(cacheWriteResults, output, pickle.HIGHEST_PROTOCOL)



# def main():
    
#     # Import your model
#     #################################################
#     """
#     A general way of importing the model, so instead of this:
#     >> from service.MaskRCNN import MaskRCNN
#     >> from service.PANet import PANet 

#     You can just specify the model name in config.py and it's
#     imported automatically using the model name.

#     """
#     from models.PANet.PANet import PANet
#     Model = PANet
#     MODEL_NAME = "PANet"
#     #MODEL_NAME = serviceConfig.modelName
#     #modulename = "service." + "models." + MODEL_NAME + "." + MODEL_NAME 
#     ## mod = __import__(modulename, fromlist=[MODEL_NAME])
#     #mod = locate(modulename)
#     #Model = getattr(mod, MODEL_NAME)
#     #################################################

#     print("MODEL:",MODEL_NAME)
#     #print("IMAGE:",os.path.basename(os.path.normpath(serviceConfig.image_path)))

#     img_ext = [".tif",".TIF",".tiff",".TIFF"]

#     if serviceConfig.from_dir is False:
#         print("IMAGE:",os.path.basename(os.path.normpath(serviceConfig.image_path)))
#         outputDir = os.path.join(os.path.dirname(serviceConfig.image_path),'{}/{}/'.format(serviceConfig.modelName, serviceConfig.modelVersion)) # "/home/ashwin/Desktop/Tripoli_1" 
#         run_process(serviceConfig,outputDir,Model)
#     else:
#         print("DIR:{}".format(os.path.basename(os.path.normpath(serviceConfig.img_dir_path))))

#         # Assert the image directory exists
#         assert os.path.isdir(serviceConfig.img_dir_path) is True
#         img_list = [i for i in os.listdir(serviceConfig.img_dir_path) if i[-4:] in img_ext] #img_dir_path = "/home/ashwin/Desktop/Stefan_Tripoli/1/"

#         outputListDir = os.path.join(os.path.dirname(serviceConfig.img_dir_path),os.path.basename(serviceConfig.img_dir_path), '{}/{}/'.format(serviceConfig.modelName, serviceConfig.modelVersion))#+ "_Results") # 1_Results ##os.path.join(os.path.dirname(serviceConfig.img_dir_path),'{}/{}/'.format(serviceConfig.modelName, serviceConfig.modelVersion))

#         if os.path.isdir(outputListDir) is False:
#             os.makedirs(outputListDir, exist_ok=True)

#         print("Number of images in directory = {} \n".format(len(img_list)))

#         for img in img_list:

#             # Overwrite image path in config.py
#             serviceConfig.image_path = os.path.join(serviceConfig.img_dir_path,img)  #img_dir_path = "/home/ashwin/Desktop/Stefan_Tripoli/1/20190413B_Tripoli_R1C1.tif"

#             print("IMAGE:",os.path.basename(os.path.normpath(serviceConfig.image_path)))

#             img_output_dir = os.path.join(outputListDir,os.path.basename(serviceConfig.image_path)[:-4])

#             # Create an outputDir for each image in img_list
#             if os.path.isdir(img_output_dir) is False:
#                 os.makedirs(img_output_dir, exist_ok=True)

#             run_process(serviceConfig,img_output_dir,Model)

#             print("\n")



# if __name__ == "__main__":
#     main()






