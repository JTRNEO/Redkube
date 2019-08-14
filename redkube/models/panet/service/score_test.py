import os
import numpy as np
import osgeo.ogr as ogr
import osgeo.osr as osr
import geopandas as gpd
import pandas as pd
import shutil
from tqdm import tqdm
from tabulate import tabulate
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
from shapely.ops import cascaded_union

geoimg = '/home/ashwin/Desktop/New_Data/Davis_Monthan/Davis_Monthan_AFB_20180814.tif'
outputDir = '/home/ashwin/Desktop/New_Data/Davis_Monthan/PANet_AirField/Test'
#geojson = ContourProcessor(geoimg, outputDir)
gtGeoJson = '/home/ashwin/Desktop/New_Data/Davis_Monthan/Davis_Monthan_AFB_20180814.tif.geojson'
cleanedGeojsonPath = "/home/ashwin/Desktop/New_Data/Davis_Monthan/PANet_AirField/IIAI_98_v5/cleanData.geojson"

mapping = True


gtMappings = {
    'Building':'Building',
    'Buildings': 'Building',
    'Other': 'Building',
    'Greenhouse': 'Building',
    'Hangar': 'Building',
    'Storage Tank': 'Building',
    'Residential': 'Building',
    'Construction': 'Building',
    'Warehouses': 'Building'
    ,
    'Cars': 'Vehicles',
    'Planes': 'Vehicles',
    'Fire Truck': 'Vehicles',
    'Pickup Trucks': 'Vehicles',
    'Semi Trucks': 'Vehicles',
    'Ambulance':'Vehicles',
    'Vehicles': 'Vehicles'
    ,
    'Planes':'Planes',
    'Civilian Passenger':'Planes',
    'Civ Jet':'Planes',
    'Civ Light Aircraft':'Planes',
    'Helicopter':'Planes',
    'Mil Bomber':'Planes',
    'Mil Fighter':'Planes'
    ,
    'Ships':'Ships',
    'Destroyer':'Ships',
    'Frigate':'Ships',
    'Cruiser':'Ships',
    'Cargo Ship':'Ships',
    'Boats':'Ships',
    'Submarines':'Ships'
    ,
    'Sports Stadium/Field':'Sports Stadium',
    'Tennis Court':'Sports Stadium', 
    'Basketball Court':'Sports Stadium', 
    'Soccer Field':'Sports Stadium', 
    'Baseball Diamond':'Sports Stadium',
    'Stadium':'Sports Stadium/Field', 
    'Athletic Track':'Sports Stadium'
    ,

    'Mil Vehicles':'Mil Vehicles',
    'Hovercraft':'Mil Vehicles',  
    'Tanks':'Mil Vehicles',
    'Self-propelled Artillery':'Mil Vehicles', 
    'Towed Artillery':'Mil Vehicles', 
    'APC':'Mil Vehicles', 
    'Fighting Veh.':'Mil Vehicles',
    'Support Vehicles':'Mil Vehicles'
    , 
    #'Missiles/Missile Systems':'Missiles/Missile Systems'
    #,
    'Train':'Train',
    'Engine':'Train',
    'Boxcar':'Train', 
    'Passenger Car':'Train',
    'Flatbed Car':'Train', 
    'Hopper':'Train', 
    'Tanker':'Train'

}

modelMappings = {
    'Building':'Building',
    'Buildings': 'Building',
    'Other': 'Building',
    'Greenhouse': 'Building',
    'Hangar': 'Building',
    'Storage Tank': 'Building',
    'Residential': 'Building',
    'Construction': 'Building',
    'Warehouses': 'Building'
    ,
    'Cars': 'Vehicles',
    'Planes': 'Vehicles',
    'Fire Truck': 'Vehicles',
    'Pickup Trucks': 'Vehicles',
    'Semi Trucks': 'Vehicles',
    'Ambulance':'Vehicles',
    'Vehicles': 'Vehicles'
    ,
    'Planes':'Planes',
    'Civilian Passenger':'Planes',
    'Civ Jet':'Planes',
    'Civ Light Aircraft':'Planes',
    'Helicopter':'Planes',
    'Mil Bomber':'Planes',
    'Mil Fighter':'Planes'
    ,
    'Ships':'Ships',
    'Destroyer':'Ships',
    'Frigate':'Ships',
    'Cruiser':'Ships',
    'Cargo Ship':'Ships',
    'Boats':'Ships',
    'Submarines':'Ships'
    ,
    'Sports Stadium/Field':'Sports Stadium',
    'Tennis Court':'Sports Stadium', 
    'Basketball Court':'Sports Stadium', 
    'Soccer Field':'Sports Stadium', 
    'Baseball Diamond':'Sports Stadium',
    'Stadium':'Sports Stadium', 
    'Athletic Track':'Sports Stadium'
    ,

    'Mil Vehicles':'Mil Vehicles',
    'Hovercraft':'Mil Vehicles',  
    'Tanks':'Mil Vehicles',
    'Self-propelled Artillery':'Mil Vehicles', 
    'Towed Artillery':'Mil Vehicles', 
    'APC':'Mil Vehicles', 
    'Fighting Veh.':'Mil Vehicles',
    'Support Vehicles':'Mil Vehicles'
    ,
    'Missiles/Missile Systems':'Missiles/Missile Systems'
    ,
    'Train':'Train',
    'Engine':'Train',
    'Boxcar':'Train', 
    'Passenger Car':'Train',
    'Flatbed Car':'Train', 
    'Hopper':'Train', 
    'Tanker':'Train'
}

#mapping=False
classes = [
            'BG','Planes', 'Civilian Passenger', 'Civ Jet', 'Civ Light Aircraft', 
            'Civ Transport', 'Mil Bomber', 'Mil Fighter', 'Mil Transport', 'Plane Engine', 
            'Ships', 'Destroyer', 'Frigate', 'Cruiser', 'Aircraft Carrier', 'Cargo Ship', 
            'Boats', 'Submarines', 'Sailing Ships/Boats', 'Tanker', 'Helicopter', 'Civilian', 
            'Military', 'Vehicles', 'Cars', 'Pickup Trucks', 'Motorcycles', 'Semi Truck', 'Bus', 
            'Ambulance', 'Fire Truck', 'Bridges', 'Pedestrian', 'Buildings', 'Mosques', 'Towers', 
            'Residential', 'Other', 'Greenhouse', 'Warehouses', 'Parking Lots', 'Air Traffic Control Tower', 
            'Runway', 'Hangar', 'Taxiways', 'Aprons', 'Helipad', 'Satellite Dish', 'Solar Panels', 'Storage Tank', 
            'Roundabout', 'Swimming Pool', 'Sports Stadium/Field', 'Tennis Court', 'Basketball Court', 
            'Soccer Field', 'Baseball Diamond', 'Stadium', 'Athletic Track', 'Rail(train)', 'Intersection/Crossroads', 
            'Shipping Container Lot', 'Shipping Containers', 'Crane', 'Construction', 'Floating', 'Gantry', 'Tower', 
            'Train', 'Engine', 'Boxcar', 'Passenger Car', 'Flatbed Car', 'Hopper', 'Tanker', 'Breakwater', 'Pier', 
            'Quay', 'Harbor', 'Drydocks', 'Floating Docks', 'Slip', 'Telephone Poles', 'Hovercraft', 'Mil Vehicles', 
            'Tanks', 'Self-propelled Artillery', 'Towed Artillery', 'APC', 'Fighting Veh.', 'Support Vehicles', 
            'Missiles/Missile Systems', 'Comms Towers', 'Power Station']
classes = classes[1:]
#print("Number of classes=",len(classes))

    

gtFrame = gpd.read_file(gtGeoJson)
if mapping is True:
    gtFrame['mlabel'] = gtFrame['Label'].apply(lambda l: gtMappings[l] if l in gtMappings else False)
    gtGroup = gtFrame[gtFrame['mlabel'] != False]
    # Fix self intersections by removing non-valid polygons
    gtGroup = gtGroup[gtGroup['geometry'].is_valid == True].groupby('mlabel')

    #print(gtFrame)

    modelFrame  = gpd.read_file(cleanedGeojsonPath).dropna(subset=['geometry'])

    modelFrame['mlabel'] = modelFrame['Label'].apply(lambda l: modelMappings[l] if l in modelMappings else False)
    modelGroup = modelFrame[modelFrame['mlabel'] != False].groupby('mlabel')
    
    # print(modelFrame)
    
    

    # print("\n")

    # print(gtGroup.groups)
    # print("\n")
    # print(modelGroup.groups)
else:
    gtFrame['mlabel'] = gtFrame['Label'].apply(lambda l: l if l in classes else False)
    gtGroup = gtFrame[gtFrame['mlabel'] != False]
    # Fix self intersections by removing non-valid polygons
    gtGroup = gtGroup[gtGroup['geometry'].is_valid == True].groupby('mlabel')

    modelFrame  = gpd.read_file(cleanedGeojsonPath).dropna(subset=['geometry'])
    modelFrame['mlabel'] = modelFrame['Label'].apply(lambda l: l if l in classes else False)
    modelGroup = modelFrame[modelFrame['mlabel'] != False].groupby('mlabel')



# print("GTGroups=",gtGroup.groups)
# print("ModelGroups=",modelGroup.groups)

# print("Number of GT groups=",len(gtGroup.groups))
# print("Number of model groups=",len(modelGroup.groups))

# for label in gtGroup.groups:
#     print(label)

# print("\n")


# for label in modelGroup.groups:
#     print(label)

#print("Amb=",gtGroup.get_group("Ambulance"))
#print("Amb=",modelGroup.get_group("Ambulance"))

rows = 3

csvLabels = []
csvIoUs = []
csvF1 = []
csvFalsePositives = []
csvFalseNegatives = []
csvObjectsInGT = []
csvObjectsInModel = []
csvObjectsIntersect = []

        
        

for label in tqdm(gtGroup.groups, desc='Calculating IoU scores'):
    csvLabels.append(label)
    print(label)

    bgt = gtGroup.get_group(label)
    csvObjectsInGT.append(len(bgt.index))

    bgm = modelGroup.get_group(label)
    csvObjectsInModel.append(len(bgm.index))

    try:
        bgm = modelGroup.get_group(label)
        csvObjectsInModel.append(len(bgm.index))
    except:
        continue



    

    hitsDf = gpd.overlay(bgm, bgt, how="intersection")
    csvObjectsIntersect.append(len(hitsDf.index))

    # errorDf = gpd.overlay(bgm, bgt, how="symmetric_difference")
    falsePositiveDf = gpd.overlay(bgm, bgt, how="difference")
    falseNegativeDf = gpd.overlay(bgt, bgm, how="difference")

    lbs = ['hits', 'falsePos', 'falseNeg']

    # NOTE:
    # If the two polygons intersect, the result of union or unary_union is a Polygon else it's a MultiPolygon

    geom = [hitsDf['geometry'].unary_union,
            falsePositiveDf['geometry'].unary_union,
            falseNegativeDf['geometry'].unary_union]

    area = [p.area for p in geom]

    totalArea = sum(area)

    hits = [area[0]/totalArea]*rows
    csvIoUs.append(hits[-1])

    falsePositives = [area[1]/totalArea]*rows
    csvFalsePositives.append(falsePositives[-1])

    falseNegatives = [area[2]/totalArea]*rows
    csvFalseNegatives.append(falseNegatives[-1])

    precision = hits[-1] / (hits[-1] + falsePositives[-1])
    recall = hits[-1] / (hits[-1] + falseNegatives[-1])
    f1 = (2 * precision * recall) / (precision + recall)
    csvF1.append(f1)


    dfData = {'label': lbs,
              'geometry': geom,
              'hit': hits,
              'falsePos': falsePositives,
              'falseNeg': falseNegatives}

    # df = pd.DataFrame.from_dict(dfData, orient='index')
    # df.transpose()
    # dfData = df.to_dict()
    # print("Label:",label)
    # print("Geometry:",geom)
    # print("DFDATA:",dfData)

    # if geom[0].geom_type == "Polygon":
    #     geom = [cascaded_union(geom)]

    # print("Geometry:",geom)

    dfData["geometry"] = [MultiPolygon([feature]) if feature.geom_type == "Polygon" else feature for feature in dfData["geometry"]]

    # print("Geometry:",geom)



    finalDf = gpd.GeoDataFrame(dfData, crs=modelFrame.crs)

    fileName = os.path.join(outputDir, '{}-{:02d}.geojson'.format(label, int(100*hits[0])))

    finalDf.to_file(driver='GeoJSON', filename=fileName)



# pdf = pd.DataFrame({
#     'Label': csvLabels,
#     'F1 Score': csvF1,
#     'IoU': csvIoUs,
#     'FalsePositive': csvFalsePositives,
#     'FalseNegative': csvFalseNegatives,
#     'ObjectsInGTruth': csvObjectsInGT,
#     'ObjectsInModel': csvObjectsInModel,
#     'ObjectsIntersect': csvObjectsIntersect
# })

a = {
    'Label': csvLabels,
    'F1 Score': csvF1,
    'IoU': csvIoUs,
    'FalsePositive': csvFalsePositives,
    'FalseNegative': csvFalseNegatives,
    'ObjectsInGTruth': csvObjectsInGT,
    'ObjectsInModel': csvObjectsInModel,
    'ObjectsIntersect': csvObjectsIntersect
}

pdf = pd.DataFrame.from_dict(a,orient='index')
#pdf = pd.DataFrame(a,orient='index')
pdf.transpose()

csvFileName = os.path.join(outputDir, 'score.csv')
pdf.to_csv(csvFileName)

print(tabulate(pdf, headers='keys', tablefmt='psql'))