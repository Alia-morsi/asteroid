import pyroomacoustics as pra
import sympy as s
import numpy as np
from random import seed, random, uniform, choice, randrange
from math import pi, sqrt, sin, cos
from copy import deepcopy
from datetime import datetime
import torchaudio
import torch
import pdb


#instead, we opted for another approach where a room factory generates a random room with random placement parameters. So, the needed configuration is for the room factory and not the room.
room_params = {
	'corners': [], 
	'speaker_loc': 0,
	'instrument_loc': 0,
	'mic_loc': 0,
	'ir60': 0, 
	'damping': None, 
	'delay': 0, 
	'directivity': 0, 
	'mic_type': 'cardioid', # still exploring this option
	'SnR': 0,
}

def point_in_circle(x_center, y_center, radius):
    #https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
    r = radius * sqrt(random())
    theta = random() * 2 * pi
    x = x_center + r * cos(theta)
    y = y_center + r * sin(theta)
    return [x, y]

def stereo_change(x_y, h):
    #for now we'll just pretend they are 2 mics placed vertically
    R = deepcopy(x_y)
    R[1] = R[1] - 0.2
    L = deepcopy(x_y)
    L[1] - L[1] + 0.2

    mic_locs = np.c_[
            R + [h],
            L + [h]  #it's not really R L, it's more top-bottom, but whatever
        ]
    return mic_locs


#To choose 3 points where their dist. do not exceed a thresh.
#TODO: set ceiling to how many times we must iterate, because it gets too much sometimes
def choose_3_points(x_center_range, y_center_range, radius, intrapoint_distance):
    x_center = uniform(x_center_range[0], x_center_range[1])
    y_center = uniform(y_center_range[0], y_center_range[1])

    point_1 = s.Point(point_in_circle(x_center, y_center, radius))
    point_2 = s.Point(point_in_circle(x_center, y_center, radius))
    #print('point 1 {}'.format(point_1))
    #print('point 2 {}'.format(point_2))

    while(not (intrapoint_distance[0] <= point_1.distance(point_2) <= intrapoint_distance[1])):
        point_2 = s.Point(point_in_circle(x_center, y_center, radius))
        #print('point 2 {}'.format(point_2))

    point_3 = s.Point(point_in_circle(x_center, y_center, radius))
    #print('point 3 {}'.format(point_3))

    while (not (intrapoint_distance[0] <= point_1.distance(point_3) <= intrapoint_distance[1])
           or not(intrapoint_distance[0] <= point_2.distance(point_3) <= intrapoint_distance[1])):
        point_3 = s.Point(point_in_circle(x_center, y_center, radius))
        #print('point 3 {}'.format(point_3))

    #convert to an array of floats
    point_1 = [float(point_1.x), float(point_1.y)]
    point_2 = [float(point_2.x), float(point_2.y)]
    point_3 = [float(point_3.x), float(point_3.y)]
    return [point_1, point_2, point_3]

class Room:
    def __init__(self, room, height, backing_track_index, instrument_track_index, kwargs):
        self.room = room
        self.backing_track_index = backing_track_index
        self.instrument_track_index = instrument_track_index
        self.height = height
        self.backing_track = None
        self.instrument_track = None
        self.backing_track_mute = True
        self.instrument_track_mute = True
        self.kwargs = kwargs

    def toggle_mute_backing_track(self):
        if self.backing_track_mute:
            self.room.sources[self.backing_track_index].signal = self.backing_track
            self.backing_track_mute = False
        else:
            self.room.sources[self.backing_track_index].signal = np.zeros_like(self.backing_track)
            self.backing_track_mute = True
        return

    def toggle_mute_instrument_track(self):
        if self.instrument_track_mute:
            self.room.sources[self.instrument_track_index].signal = self.instrument_track
            self.instrument_track_mute = False
        else:
            self.room.sources[self.instrument_track_index].signal = np.zeros_like(self.instrument_track)
            self.instrument_track_mute = True
        return

    def add_instrument_track(self, signal):
        # Had to convert it to mono, because it seems that the generated IR expects mono whereas my audio is stereo
        # the format expected is that of scipy.io wavfile
        # if that will change, adapt them somehow.
        self.instrument_track = signal[0]
        self.room.sources[self.instrument_track_index].signal = np.zeros_like(self.instrument_track)
        self.instrument_track_mute = True

    def add_backing_track(self, signal):
        #TODO: Add Stereo Later. Probably that means i'll have to add 2 sources instead of 1
        self.backing_track = signal[0]
        self.room.sources[self.backing_track_index].signal = np.zeros_like(self.backing_track)
        self.backing_track_mute = True

    def read_mic_output(self):
        #read from both the mics, since we placed 2 for a stereo output
        #pdb.set_trace()
        self.room.image_source_model()
        self.room.simulate()
        return self.room.mic_array.signals[0, :], self.room.mic_array.signals[1, :]
        #the way the mic output is indexed is different than how we index the sources 
        #because it's a different sound object
        #return torch.vstack([torch.FloatTensor(self.room.mic_array.signals[0,:]),
        #                     torch.FloatTensor(self.room.mic_array.signals[1,:])])

    def show_room(self):
        #these are just 'educated-guess' dimensions that will prob.
        #cover whatever room we end up constructing.
        fig, ax = self.room.plot()
        ax.set_xlim([-1, 10])
        ax.set_ylim([-1, 10])
        ax.set_zlim([-1, 5]);

    def get_coordinates(self):
        return {
            'microphone': self.room.mic_array.center,
            'backing_track': self.room.sources[self.backing_track_index].position,
            'instrument': self.room.sources[self.instrument_track_index].position,
        }
    def get_other_parameters(self):
        return self.kwargs

    def get_directivity(self):
        #TODO: make sure factory is adding reasonable directivity parameters.
        return {
            'microphone': self.room.room_mic_array.directivity
        }
    def get_dimensions(self):
        #TODO: complete with the actual stuff.
        corners = np.array([wall.corners[:, 0] for wall in self.room.walls]).T
        zipped_lists = zip(corners[0], corners[1])
        sorted_pairs = sorted(zipped_lists)
        return {
            'room_type': True, #probably wont matter, if we'll pass dim as corners
            'corners': sorted_pairs,
            'height': self.height
        }


class RoomFactory:
    def __init__(self, shoebox_units, room_units, room_units_maxes, room_fixed, room_fixed_maxes, room_heights, 
                 room_mult_factor, mic_placement_heights, source_placement_heights, placement_margins, 
                 circle_radii, intrapoint_distance_min, intrapoint_distance_max, materials, max_order):
        self.shoebox_units = shoebox_units
        self.room_units = room_units
        self.room_units_maxes = room_units_maxes
        self.room_fixed = room_fixed
        self.room_fixed_maxes = room_fixed_maxes
        self.room_heights = room_heights
        self.room_mult_factor = room_mult_factor
        self.mic_placement_heights = mic_placement_heights
        self.source_placement_heights = source_placement_heights
        self.placement_margins = placement_margins
        self.circle_radii = circle_radii
        self.intrapoint_distance_min = intrapoint_distance_min 
        self.intrapoint_distance_max = intrapoint_distance_max
        self.materials = materials
        self.max_order = max_order
        
    def create_room(self):
        room_options = ['shoebox_units', 'room_units', 'room_fixed']
        seed(datetime.now())
       
        #choose material, for now just a single choice for the whole room
        material_key = choice(self.materials)
        m = pra.Material(material_key)

        kwargs = {
            'materials': m,
            'air_absorption': choice([True, False]),
            'ray_tracing': False, #choice([True, False]),
            'max_order': randrange(self.max_order[0], self.max_order[1])
        }
         
        #print(kwargs)

        #choose type of room
        room_type = choice(room_options)
        if room_type == 'room_fixed':
            unit_ratio = 1
        else:
            unit_ratio = uniform(self.room_mult_factor[0], self.room_mult_factor[1])

        radius = uniform(self.circle_radii[0], self.circle_radii[1])
        
        #get chosen room type with getattr
        #room = pra.ShoeBox(unit_ratio * np.array(choice(getattr(self, room_type))), fs=44100, **kwargs
        #                    ) if room_type == 'shoebox_units' else pra.Room.from_corners(
        #                    (unit_ratio * np.array(choice(getattr(self, room_type)))).T, fs=44100, **kwargs)
        

        #set height:
        height = choice(self.room_heights)
        dimensions = unit_ratio * np.array(choice(getattr(self, room_type)))

        if room_type == 'shoebox_units':
            dimensions = np.append(dimensions, height) 
            room = pra.ShoeBox(dimensions, fs=44100, **kwargs)
        else:
            room = pra.Room.from_corners(dimensions.T, fs=44100, **kwargs) 
            #room.extrude(height)
            room.extrude(height, materials=m)

        #add material key to kwargs, since it was already used as a constructor and now we can add whatever we want.
        kwargs['material_key'] = material_key
        print(kwargs) 
        #pdb.set_trace()
        if isinstance(room, pra.room.ShoeBox):
            max_x = room.shoebox_dim[0]
            max_y = room.shoebox_dim[1]
        
        else:
            #just read max_x and max_y from the parameters. 
            #Currently each room type has the same max x and y range. 
            max_x, max_y = getattr(self, '{}_maxes'.format(room_type))
            max_x = max_x * unit_ratio
            max_y = max_y * unit_ratio
            
        #These will not work if the radius is too big compared to the maxes. 
        #If this happens, choose another radius
        while(radius+self.placement_margins >= max_y or 
              radius+self.placement_margins >= max_x or
              max_x - self.placement_margins - radius < self.placement_margins + radius or
              max_y - self.placement_margins - radius < self.placement_margins + radius):
            radius = uniform(self.circle_radii[0], self.circle_radii[1])

        x_center_range = (self.placement_margins + radius, 
                         max_x - self.placement_margins - radius)
        y_center_range = (self.placement_margins + radius,
                         max_y - self.placement_margins - radius)
    
        placement_points = choose_3_points(x_center_range, y_center_range, radius, 
                        (self.intrapoint_distance_min, self.intrapoint_distance_max))

        #put the sources and mic
        room.add_source(placement_points[0] + [choice
                    (self.source_placement_heights)])
        room.add_source(placement_points[1] + [choice
                    (self.source_placement_heights)])
        
        mic_locs = stereo_change(placement_points[2], 
                        choice(self.mic_placement_heights))
        
        room.add_microphone_array(mic_locs)
        
        #materials = pra.Material(e_absorption)
        #where e_absorption is set from the Sabine Target and the room dimensions.
        #Also, later add mic directivity
        
        return Room(room, height, backing_track_index=0, instrument_track_index=1, kwargs=kwargs)
