import os
import numpy as np
import osgeo.ogr as ogr, osr
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point
import pdb

class ContourProcessor():
    def __init__(self, geoImg, outputDir):

        self.geoImg = geoImg

        mercator = osr.SpatialReference()
        # convert everything to meters for clearer area, length, width operations
        mercator.ImportFromEPSG(3857)
        self.projTransform = osr.CoordinateTransformation(self.geoImg.tifProjection, mercator)

        driver = ogr.GetDriverByName("GeoJSON")
        self.outputDir = outputDir
        self.fullDataGeojsonPath = os.path.join(outputDir, 'fullData.geojson')
        if os.path.exists(self.fullDataGeojsonPath): os.remove(self.fullDataGeojsonPath)
        self.fullDataSource = driver.CreateDataSource(self.fullDataGeojsonPath)
        # create the layer
        self.fullDataLayer = self.fullDataSource.CreateLayer("payload", mercator, ogr.wkbPolygon)
        # Add the fields we're interested in
        field_type = ogr.FieldDefn("Label", ogr.OFTString)
        field_type.SetWidth(24)
        self.fullDataLayer.CreateField(field_type)
        field_type = ogr.FieldDefn("ObjId", ogr.OFTString)
        field_type.SetWidth(24)
        self.fullDataLayer.CreateField(field_type)
        self.fullDataLayer.CreateField(ogr.FieldDefn("Area", ogr.OFTInteger))
        self.fullDataLayer.CreateField(ogr.FieldDefn("ClassId", ogr.OFTInteger))
        self.fullDataLayer.CreateField(ogr.FieldDefn("Score", ogr.OFTReal))
        self.fullDataLayer.CreateField(ogr.FieldDefn("is_partial", ogr.OFTInteger))

        self.cleanedGeojsonPath = os.path.join(outputDir, 'cleanData.geojson')
        self.cleanedBboxGeojsonPath = os.path.join(outputDir, 'cleanData_bbox.geojson')

        self.patchesGeojsonPath = os.path.join(outputDir, 'patchesData.geojson')
        if os.path.exists(self.patchesGeojsonPath): os.remove(self.patchesGeojsonPath)
        self.patchesSource = driver.CreateDataSource(self.patchesGeojsonPath)
        # create the layer
        self.patchesLayer = self.patchesSource.CreateLayer("payload", mercator, ogr.wkbPolygon)

    def addPatchBoundary(self, left, up):
        size = self.geoImg.subsize - 1
        bbox = np.array([[0, 0], [0, size], [size, size], [size, 0]], np.float)
        feature = ogr.Feature(self.patchesLayer.GetLayerDefn())
        bbox[::, ::] += (float(left), float(up))
        feature.SetGeometry(self.reproject(bbox))
        self.patchesLayer.CreateFeature(feature)

        buf = 5  # bufferForPartialDetection
        bboxPartialDetection = np.array([[buf, buf], [buf, size - buf], [size - buf, size - buf], [size - buf, buf]],
                                        np.float)
        bboxPartialDetection[::, ::] += (float(left), float(up))
        return self.reproject(bboxPartialDetection)


    def addFeature(self, left, up, modelInference, patchGeom):
        score = modelInference['score']
        classId = modelInference['classId']
        label = modelInference['label']
        polyVerts = modelInference['verts']

        feature = ogr.Feature(self.fullDataLayer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("Label", label)
        feature.SetField("ClassId", int(classId))
        feature.SetField("Score", float(score))
        feature.SetField("ObjId", modelInference['objId'])


        polyVerts[::, ::] += (int(left), int(up))
        featureGeom = self.reproject(polyVerts)
        # we have mercator reprojection so area will be always in sq Meters
        feature.SetField("Area", featureGeom.Area())
        feature.SetGeometry(featureGeom)

        feature.SetField("is_partial", 0 if featureGeom.Within(patchGeom) else 1)
        # Create the feature in the layer (geojson)
        self.fullDataLayer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    def cleanUp(self):
        # save and close patch and fulldata shape files
        self.fullDataSource = None
        self.patchesSource = None

        # load full data file in a dataframe
        gdf = gpd.GeoDataFrame.from_file(self.fullDataGeojsonPath)
        gdf.crs = {'init': 'epsg:3857'}
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.is_partial == 0]

        assert gdf.empty is False, "GeoDataFrame is empty"

        itemsToRemove = {}

        # resolve conflicts inside same Labeled groups
        for label, labelDf in tqdm(gdf.groupby('Label'), desc='Processing groups'):
            intersectGroups = gpd.sjoin(labelDf, labelDf, how="inner", op='intersects').groupby('ObjId_left')
            labelDf = labelDf.set_index(['ObjId']).sort_values(by=['Score'], ascending=False)

            for objId in tqdm(labelDf.index, desc='Processing {}'.format(label)):
                if objId in itemsToRemove: continue
                #pdb.set_trace()
                try:
                    idf = intersectGroups.get_group(objId)
                except KeyError:
                    continue
                #if idf is None: continue

                bestPoly = labelDf.loc[objId]

                for index, poly in idf.iterrows():
                    # ignore self intersection
                    if poly['ObjId_left'] == poly['ObjId_right']: continue

                    id = poly['ObjId_right']
                    if id not in itemsToRemove:
                        # do not consider annotations intersecting on less then 30% as conflict
                        if labelDf.loc[id].geometry.intersection(bestPoly.geometry).area / bestPoly.geometry.union(
                                labelDf.loc[id].geometry).area > 0.3:
                            if id == 'left-7416_up-9064_obj-60':
                                print(idf)
                            itemsToRemove[id] = True

        gdf = gdf.set_index(['ObjId']).drop(itemsToRemove.keys())

        # gdf['geometry'] = gdf['geometry'].simplify(2)

        if os.path.exists(self.cleanedGeojsonPath): os.remove(self.cleanedGeojsonPath)
        gdf.to_file(driver='GeoJSON', filename=self.cleanedGeojsonPath)

        if os.path.exists(self.cleanedBboxGeojsonPath): os.remove(self.cleanedBboxGeojsonPath)
        gdf.geometry = gdf.geometry.apply(lambda p: p.minimum_rotated_rectangle)

        def getRectangleSide(polygon, width=True):
            ps = list(map(lambda p: Point(p), polygon.exterior.coords))

            return max([ps[0].distance(ps[1]), ps[1].distance(ps[2])]) if width else min(
                [ps[0].distance(ps[1]), ps[1].distance(ps[2])])

        gdf['width'] = gdf.geometry.apply(lambda p: getRectangleSide(p))
        gdf['height'] = gdf.geometry.apply(lambda p: getRectangleSide(p, width=False))

        gdf.to_file(driver='GeoJSON', filename=self.cleanedBboxGeojsonPath)

    def reproject(self, relativePolygon):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        xoffset, px_w, rot1, yoffset, rot2, px_h = self.geoImg.afineTrnsform

        for x, y in relativePolygon:
            posX = px_w * x + rot1 * y + xoffset
            posY = rot2 * x + px_h * y + yoffset

            # shift to the center of the pixel
            posX += px_w / 2.0
            posY += px_h / 2.0

            ring.AddPoint_2D(posX, posY)

        x, y = ring.GetPoint_2D(0)
        ring.AddPoint_2D(x, y)
        ring.Transform(self.projTransform)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # print(poly.ExportToWkt())

        # Simplify Polygon. If Simplify fails, try SimplifyPreserveTopology()

        sim_poly = poly.Simplify(0.5)

        if sim_poly.IsEmpty() is True:
            # print("Empty Polygon! Opting for SimplifyPreserveTopology()")
            sim_poly = poly.SimplifyPreserveTopology(0.5)

            # Confirm resulting polygon is not empty
        assert sim_poly.IsEmpty() is False

        return sim_poly

    def deletePartialObjects(self, gdf):
        patchDf = gpd.read_file(self.patchesGeojsonPath)
        newDf = gpd.sjoin(gdf, gpd.GeoDataFrame(patchDf), op='within')
        # newDf = None
        # for index, df in patchDf.iterrows():
        #     wdf = gpd.sjoin(gdf, gpd.GeoDataFrame(df), op='within')
        #     newDf = wdf if newDf is None else newDf.append(wdf)
        print(newDf.columns)
        print(newDf.head())
        return newDf
