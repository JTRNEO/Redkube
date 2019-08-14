import os

import numpy as np
import cv2
import tifffile as tiff
import copy
from osgeo import gdal, osr, ogr
import os

class GeoImage():
    def __init__(self,
                 imgPath,
                 gap=100,
                 subsize=1024):
        self.imgPath = imgPath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.imgPath = imgPath
        self.splits = []
        #self.BuildSplits(1)

        ds = gdal.Open(imgPath)
        self.afineTrnsform = ds.GetGeoTransform()
        self.tifProjection = osr.SpatialReference(wkt=ds.GetProjection())
        self.pixelSize = (self.afineTrnsform[1],-self.afineTrnsform[5])

        del ds

    def getProjection(self):
        return self.tifProjection

    def getCv2ImgFromSplit(self, xyOffset):
        left, up = xyOffset
        subimg = copy.deepcopy(self.resizeimg[up: (up + self.subsize), left: (left + self.subsize)])
        return subimg

    def BuildSplits(self, rate):
        img = tiff.imread(self.imgPath)
        assert np.shape(img) != ()

        if (rate != 1):
            self.resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            self.resizeimg = img

        weight = np.shape(self.resizeimg)[1]
        height = np.shape(self.resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)

                self.splits.append([left, up])

                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def getSplits(self): return self.splits

    def convertToJpeg(self, jpegPath):
        ds = gdal.Open(self.imgPath)
        ds_jpeg = gdal.Translate(jpegPath, ds, noData =0, format='JPEG')
        dir = os.path.dirname(jpegPath)
        file = os.path.basename(jpegPath)
        os.rename(os.path.join(dir,'{}.aux.xml'.format(file)), os.path.join(dir,'geoInfo.xml'))
        del ds
        del ds_jpeg


if __name__ == '__main__':
    geoImg = GeoImage(r'/home/kirill/Downloads/GeoData/Davis_Monthan/Davis_Monthan_AFB_20180814.tif')
    print(geoImg.getSplits())
