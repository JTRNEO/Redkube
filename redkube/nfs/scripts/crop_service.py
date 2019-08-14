import cv2
import random
import numpy as np
import os
import redis
import rediswq
#from getfile import get_image_size,get_stride,get_image_path,get_image_bs
import argparse
import tifffile as tiff
#import yaml

def parse_args():
    parser=argparse.ArgumentParser(description='inference')
    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--image_dir',type=str,required=True)
    parser.add_argument('--image_name',type=str,required=False)
    parser.add_argument('--stride',required=False,type=int,default=1024)
    parser.add_argument('--image_size',required=True,type=int,default=1024)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--yaml',required=False)
    parser.add_argument('--pods',required=True)
    args=parser.parse_args()
    return args



def get_splits(args, img):


        gap = 200
        splits = []

        weight = np.shape(img)[1]
        height = np.shape(img)[0]

        slide = args.image_size - gap

        left, up = 0, 0
        while (left < weight):
            if (left + args.image_size >= weight):
                left = max(weight - args.image_size, 0)
            up = 0
            while (up < height):
                if (up + args.image_size >= height):
                    up = max(height - args.image_size, 0)

                splits.append([left, up])

                if (up + args.image_size >= height):
                    break
                else:
                    up = up + slide
            if (left + args.image_size >= weight):
                break
            else:
                left = left + slide

        return splits




def crop_service():

        r = redis.Redis(host='redis', port=6379, decode_responses=True)

        args = parse_args()

        
        #with open(args.yaml) as f:
        #    yam = yaml.load(f)
        
        # ########### Multi Image Support ######################
        # img_dir = args.image_dir # Same as serviceconfig image dir
        # img_ext = [".tif",".TIF",".tiff",".TIFF"]

        # # Assert the image directory exists
        # assert os.path.isdir(img_dir) is True
        # img_list = [i for i in os.listdir(img_dir) if i[-4:] in img_ext]

        # crop_store_dir = os.path.join("/mnt","crops")
        # if os.path.isdir(crop_store_dir) is False:
        #     print("Creating Crop Directory")
        #     os.mkdir(crop_store_dir)


        # imid = 1

        # for img_name in sorted(img_list):
        #     print("Making crops for ",img_name)
        #     img_path = os.path.join(img_dir,img_name)
        #     img = tiff.imread(img_path)
        #     splits = get_splits(args, img)
        #     assert np.shape(img) != ()
        #     #img_name = os.path.basename(os.path.normpath(img_path))
        #     img_crop_dir = os.path.join(crop_store_dir,str(imid))
        #     if os.path.isdir(img_crop_dir) is False:
        #         # Make directory to store image crops
        #         os.mkdir(img_crop_dir)

        #     # Save crops into crops folder
        #     for i in splits:
        #         left, up = i[0], i[1]
        #         crop = img[up: (up + args.image_size), left: (left + args.image_size)]
        #         img_crop_path = os.path.join(img_crop_dir,'{}_{}_{}.png'.format(imid,i[0],i[1]))
        #         cv2.imwrite(img_crop_path,cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        #     # Send crop batches to queue
        #     if len(splits) % args.batch_size == 0:
        #       num_batches = len(splits) // args.batch_size
        #       for i in range(num_batches):
        #         batch = splits[i * args.batch_size : i * args.batch_size + args.batch_size]
        #         batch[0].insert(0,imid) # [[left, up, imid]]
        #         r.rpush('job',str(batch))
        #     else:
        #       num_dummies = args.batch_size - (len(splits) % args.batch_size)
        #       n = len(splits)
        #       for i in range(num_dummies):
        #         dummy_img = np.zeros((args.image_size,args.image_size,3),dtype=np.uint8)
        #         dummy_img = np.asarray(dummy_img,'f')
        #         n += 1
        #         cv2.imwrite(img_path, cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))

        #       num_batches = n // args.batch_size
        #       for i in range(num_batches):
        #         batch = splits[i * args.batch_size : i * args.batch_size + args.batch_size]
        #         r.rpush('job',str(batch))

        #     # Flag to show separate image
        #     r.rpush('job',1)

        #     imid += 1
            

        # ######################################################

        img = tiff.imread(args.dataset)
        img_dir = args.image_dir
        
        assert np.shape(img) != ()

        splits = get_splits(args, img)
        crop_store_dir = os.path.join("/mnt","crops")
        #crop_store_dir = os.path.join("/home/ubuntu/sniperkube/service_nfs/sat_images","crops")
        if os.path.isdir(crop_store_dir) is False:
            print("Creating Crop Directory")
            os.mkdir(crop_store_dir)

        # Save crops into crops folder
        for i in splits:
            left, up = i[0], i[1]
            crop = img[up: (up + args.image_size), left: (left + args.image_size)]
            img_path = os.path.join(crop_store_dir,'{}_{}.png'.format(i[0],i[1]))
            cv2.imwrite(img_path,cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # Send crop batches to queue
        if len(splits) % args.batch_size == 0:
          num_batches = len(splits) // args.batch_size
          for i in range(num_batches):
            batch = splits[i * args.batch_size : i * args.batch_size + args.batch_size]
            r.rpush('job',str(batch))
        else:
          num_dummies = args.batch_size - (len(splits) % args.batch_size)
          n = len(splits)
          for i in range(num_dummies):
            dummy_img = np.zeros((args.image_size,args.image_size,3),dtype=np.uint8)
            dummy_img = np.asarray(dummy_img,'f')
            n += 1
            cv2.imwrite(img_path, cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))

          num_batches = n // args.batch_size
          for i in range(num_batches):
            batch = splits[i * args.batch_size : i * args.batch_size + args.batch_size]
            r.rpush('job',str(batch))
        

        #with open(args.yaml) as f:
        #   yam = yaml.load(f)

        # Number of pods
        #pods = yam['spec']['parallelism']
        # Number of GPUs per pod
        #gpus_per_pods = yam['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu']

        #pods = 10
        pods = int(args.pods)
        for i in range(pods - 1):
            r.rpush('status', 1)


        print("Done!")






if __name__ == '__main__':
    crop_service()

