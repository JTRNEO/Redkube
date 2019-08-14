# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_spacenet_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'building']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds


def get_ships_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'ship']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_abu_musa_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 
        'Underground Shelter',
        'Communication Tower',
        'Dense Structures',
        'Vehicle',
        'Cargo container',
        'Ship',
        'Swimming pool',
        'Sports field',
        'Storage Tank',
        'Standalone Building',
        'Defensive Earthworks']
        
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds


def get_iiai_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    
    # classes = [
    #     '__background__','Planes', 'Civilian Passenger', 'Civ Jet', 'Civ Light Aircraft', 
    #     'Civ Transport', 'Mil Bomber', 'Mil Fighter', 'Mil Transport', 'Plane Engine', 
    #     'Ships', 'Destroyer', 'Frigate', 'Cruiser', 'Aircraft Carrier', 'Cargo Ship', 
    #     'Boats', 'Submarines', 'Sailing Ships/Boats', 'Tanker', 'Helicopter', 'Civilian', 
    #     'Military', 'Vehicles', 'Cars', 'Pickup Trucks', 'Motorcycles', 'Semi Truck', 'Bus', 
    #     'Ambulance', 'Fire Truck', 'Bridges', 'Pedestrian', 'Buildings', 'Mosques', 'Towers', 
    #     'Residential', 'Other', 'Greenhouse', 'Warehouses', 'Parking Lots', 'Air Traffic Control Tower', 
    #     'Runway', 'Hangar', 'Taxiways', 'Aprons', 'Helipad', 'Satellite Dish', 'Solar Panels', 'Storage Tank', 
    #     'Roundabout', 'Swimming Pool', 'Sports Stadium/Field', 'Tennis Court', 'Basketball Court', 
    #     'Soccer Field', 'Baseball Diamond', 'Stadium', 'Athletic Track', 'Rail(train)', 'Intersection/Crossroads', 
    #     'Shipping Container Lot', 'Shipping Containers', 'Crane', 'Construction', 'Floating', 'Gantry', 'Tower', 
    #     'Train', 'Engine', 'Boxcar', 'Passenger Car', 'Flatbed Car', 'Hopper', 'Tanker', 'Breakwater', 'Pier', 
    #     'Quay', 'Harbor', 'Drydocks', 'Floating Docks', 'Slip', 'Telephone Poles', 'Hovercraft', 'Mil Vehicles', 
    #     'Tanks', 'Self-propelled Artillery', 'Towed Artillery', 'APC', 'Fighting Veh.', 'Support Vehicles', 
    #     'Missiles/Missile Systems', 'Comms Towers', 'Power Station'
    # ]

    # classes = ['__background__',
    #        'Planes', 
    #        'Ships', 
    #        'Helicopter', 
    #        'Vehicles', 
    #        'Bridges', 
    #        'Pedestrian', 
    #        'Buildings', 
    #        'Parking Lots', 
    #        'Airports', 
    #        'Satellite Dish', 
    #        'Solar Panels', 
    #        'Storage Tank', 
    #        'Roundabout', 
    #        'Swimming Pool',
    #        'Sports Stadium/Field',
    #        'Rail(train)', 
    #        'Intersection/Crossroads', 
    #        'Shipping Container Lot', 
    #        'Shipping Containers', 
    #        'Crane',
    #        'Train', 
    #        'Port' ,
    #        'Telephone Poles', 
    #        'Hovercraft', 
    #        'Mil Vehicles',
    #        'Missiles/Missile Systems', 
    #        'Comms Towers', 
    #        'Power Station'
    # ]

    # classes = ['__background__', 
    #            'Planes', 
    #            'Ships', 
    #            'Helicopter', 
    #            'Vehicles', 
    #            'Bridges', 
    #            'Buildings', 
    #            'Parking Lots', 
    #            'Satellite Dish', 
    #            'Solar Panels', 
    #            'Storage Tank', 
    #            'Swimming Pool', 
    #            'Sports Stadium/Field', 
    #            'Shipping Containers', 
    #            'Crane', 'Train', 
    #            'Mil Vehicles', 
    #            'Missiles/Missile Systems', 
    #            'Comms Towers']

    #12
    classes = ['__background__', 
              'Planes', 
              'Ships', 
              'Helicopter', 
              'Vehicles', 
              'Buildings', 
              'Parking Lots',  
              'Storage Tank', 
              'Swimming Pool', 
              'Sports Stadium/Field', 
              'Shipping Containers', 
              'Crane', 
              'Comms Towers']






    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

