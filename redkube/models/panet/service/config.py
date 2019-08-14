from enum import Enum

class Cache(Enum):
    none = 1
    read = 2
    write = 3

class serviceConfig:
    cacheStrategy = Cache.write
    # cacheStrategy = Cache.read
    modelName = "PANet"
    modelVersion = "Results"
    modelStorePath = '/path/to/model/MaskRCNN/V?/'

    from_dir = False

    #image_path = '/home/ashwin/Desktop/New_Data/sat_images/Aden_Port.tif'
    #image_path = '/home/ashwin/Desktop/New_Data/Davis_Monthan/Davis_Monthan_AFB_20180814.tif'
    image_path = "/mnt/sat_images/DG_Satellite_AZ_Airfield_20180818.tif"
    #image_path = "/workspace/PANet/Satellite_Images/DG_Satellite_NM_Airfield_20171121.tif"#"../Satellite_Images/DG_Satellite_NM_Airfield_20171121.tif"
    #image_path = '/home/ashwin/Desktop/New_Data/AZ_Airfield/DG_Satellite_AZ_Airfield_20180818.tif'
    #image_path = '/home/ashwin/Desktop/New_Data/sat_images/Beijing_Airport.tif'
    #image_path = '/home/ashwin/Desktop/New_Data/sat_images/DG_Satellite_Rotterdam_Port_East_20180306.tif'
    #image_path = '/home/ashwin/Desktop/New_Data/sat_images/DG_Satellite_Singapore_Port_West_20180215_B1.tif'
    #image_path = '/home/ashwin/Desktop/New_Data/sat_images/Bandar_Abbas.tif'

    img_dir_path = "/home/ashwin/Desktop/Stefan_Tripoli/1/"

    groundTruthGeoJson = None
    #groundTruthGeoJson = '/home/ashwin/Desktop/New_Data/Davis_Monthan/Davis_Monthan_AFB_20180814.tif.geojson'
    #groundTruthGeoJson = '/home/ashwin/Desktop/New_Data/sat_images/Bandar_Abbas.geojson'
    #groundTruthGeoJson = '/home/ashwin/Desktop/New_Data/sat_images/Abu_Musa_20181103.geojson'


    # gtMappings = {
    #         'Building':'Building',
    #         'Buildings': 'Building',
    #         'Other': 'Building',
    #         'Greenhouse': 'Building',
    #         'Hangar': 'Building',
    #         'Storage Tank': 'Building',
    #         'Residential': 'Building',
    #         'Construction': 'Building',
    #         'Warehouses': 'Building'
    # }

    # modelMappings = {
    #         'Building':'Building'
    #     }

    # gtMappings = {
    #         'Standalone Building':'Building',
    #         'Dense Structures': 'Building',

    #         'Ship':'Ships',
    #         'Submarine':'Ships',
    #         'Hovercraft':'Ships',

    #         'Plane':'Planes',
    #         'Helicopter':'Planes'

    # }

    # modelMappings = {
    #         'Building':'Building',
    #         'Buildings': 'Building',
    #         'Other': 'Building',
    #         'Greenhouse': 'Building',
    #         'Hangar': 'Building',
    #         'Storage Tank': 'Building',
    #         'Residential': 'Building',
    #         'Construction': 'Building',
    #         'Warehouses': 'Building'
    #         ,
    #         'Cars': 'Vehicles',
    #         'Planes': 'Vehicles',
    #         'Fire Truck': 'Vehicles',
    #         'Pickup Trucks': 'Vehicles',
    #         'Semi Trucks': 'Vehicles',
    #         'Ambulance':'Vehicles',
    #         'Vehicles': 'Vehicles'
    #         ,
    #         'Planes':'Planes',
    #         'Civilian Passenger':'Planes',
    #         'Civ Jet':'Planes',
    #         'Civ Light Aircraft':'Planes',
    #         'Helicopter':'Planes',
    #         'Mil Bomber':'Planes',
    #         'Mil Fighter':'Planes'
    #         ,
    #         'Ships':'Ships',
    #         'Destroyer':'Ships',
    #         'Frigate':'Ships',
    #         'Cruiser':'Ships',
    #         'Cargo Ship':'Ships',
    #         'Boats':'Ships',
    #         'Submarines':'Ships'
    #     }

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
