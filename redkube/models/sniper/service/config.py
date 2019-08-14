from enum import Enum

class Cache(Enum):
    none = 1
    read = 2
    write = 3

class serviceConfig:
    cacheStrategy = Cache.write

    modelName = "SNIPER"
    modelVersion = "Results"
    modelStorePath = '/path/to/model/MaskRCNN/V?/'

    from_dir = False


    #image_path = '/home/dingjian/Documents/vis_results/clint_region/Extra_Experiment_TIF/cropped_10_render.tif'
    #image_path = '/sniper/data/sniper_infer/DG_Satellite_NM_Airfield_20171121.tif'
    image_path = '/mnt/sat_images/Davis_Monthan_AFB_20180814.tif'
    #img_dir_path = '/mnt/sat_images/'
    groundTruthGeoJson = None
    #groundTruthGeoJson = '/home/dingjian/Documents/Sniper/Davis_Monthan_AFB_20180814.tif.geojson'
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
        }
