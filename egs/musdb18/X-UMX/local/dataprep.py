from pathlib import Path
import argparse
import torch.utils.data
import torchaudio
import random
import torch
import tqdm
import soundfile as sf
import copy
from scipy import signal
import os
import librosa

#additions for the ir convolutions
import pandas as pd
import librosa
import numpy as np
import torchaudio.functional.filtering as taf
from functools import partial
from scipy.io import wavfile
import pyroomacoustics as pra
from room_simulator import RoomFactory, Room
import pydub
import pdb
'''
    Folder Structure:
        >>> #train/<songname>/<variant_num>/<mic_id>/vocals.wav ---------|
        >>> #train/<songname>/<variant_num>/<mic_id>drums.wav ----------+--> input (mix), output[target]
        >>> #train/1/bass.wav -----------|
        >>> #train/1/other.wav ---------/
        Variant represents a given room + placement, mic_id depends on mic placement. We'll have 2 mics for each room
'''

parser = argparse.ArgumentParser()


#Note: Currently Unused, and not properly tested
def filter_factory(variant, args):
#   #TODO: args should taken from the config file. but for now we'll probably pass them from the calling function.
    #TODO: put more sensible cutoff frequency ranges.
    #this function should return a filter function depending on the input, and should set the parameters so that we would just need to pass in the audio on these returned functions
    return partial(taf.lowpass_bisquad, sample_rate=44100, cutoff_freq=10000)

#Note: Currently Unused, and not properly tested
def apply_ir(ir_paths, x):
    #maybe here I should make some choice based on a better random, because this will have the same path everytime
    # x is from sf.read and not librosa.load, but by the time it gets passed to this function is should have been already transposed and put in torch.audio 
    
    random_index = random.randint(0, len(ir_paths))
    ir_path = ir_paths[random_index]

    while(not ir_path.suffixes[-1] == '.wav'): #this could also be causing slowness
        ir_path = ir_paths[random.randint(0, len(ir_paths))]

    room_ir, fs = librosa.load(ir_path, sr=44100) #The Room IR is loaded as Mono

    room_ir = torch.tensor(x_ir, dtype=torch.float)

    #convert x_ir into stereo
    x_ir = torch.vstack([x_ir, x_ir])
    x = x.reshape(1, x.shape[0], x.shape[1])
    x_ir = x_ir.reshape((x_ir.shape[0], 1, x_ir.shape[1]))

    conv_result = torch.nn.functional.conv1d(x, x_ir, groups=2)
    conv_result = conv_result.reshape(conv_result.shape[1], conv_result.shape[2])

    return conv_result, ir_path 
   

def normalize(signal, rms_level=0):

    """
    Copied from https://superkogito.github.io/blog/2020/04/30/rms_normalization.html
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    # linear rms level and scaling factor
    r = 10**(rms_level / 10.0)
    a = np.sqrt( (len(signal) * r**2) / np.sum(signal**2) )

    # normalize
    y = signal * a

    return y

def normalize_and_convert(channels):
    channel1 = torch.FloatTensor(librosa.util.normalize(channels[0]))
    channel2 = torch.FloatTensor(librosa.util.normalize(channels[1]))

    return torch.vstack([channel1, channel2])



#create a dictionary of corners which we can use for defining the rooms
#expects input in the form of the 
def simulate_room(room_params, backing_track, main_stem):
    #maybe just a slight modification to the dictionary and then making the room through a kwargs**
    room
    
    #make sure that sample rates are consistent and documented
    #be sure to get the output in stereo again, and putting summing it with torchaudio before returning
    return degraded_audio_mix, degraded_target_stem #since at this point, it might be worth a discussion which output is better, the clean stem or the degraded stem
    

"""
Args:
    root is the root path of the dataset
"""
class MUSDB18LeakageDataGenerator():
    def __init__(
        self,
        clean_train_data,
        clean_test_data, 
        output_train_data,
        output_test_data,
        ir_paths=None,
        sources=["vocals", "bass", "drums", "other"],
        targets="vocals", #can be changed to anything.. 
        suffix=".wav",
        samples_per_track=1,
        random_segments=False,
        split='test',
        random_track_mix=False,
        sample_rate=44100,
        room_factory=None
    ):

        self.clean_train_data = Path(clean_train_data).expanduser()
        self.clean_test_data = Path(clean_test_data).expanduser()
        self.output_train_data = Path(output_train_data).expanduser()
        self.output_test_data = Path(output_test_data).expanduser()

        self.ir_paths = ir_paths 
        self.sample_rate = sample_rate
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.samples_per_track = samples_per_track
        self.variants = 10 # number of data variants we should create for every song.
        self.split = split
        self.tracks = list(self.get_tracks())
        #self.speaker_irs = list(self.get_irs('speaker'))
        #self.room_and_mic_irs = list(self.get_irs('room_and_mic'))
        self.room_factory = room_factory
        #self.filter_irs = list(self.get_irs('filter'))
        if not self.tracks:
            raise RuntimeError("No tracks found.")


    def process_audio(self, snr, filter_func, audio_mix, clean_backing_track, audio_sources, targets_list, covered_targets):
        #filter the backing track to give a sense of the loudspeaker
        processed_backing_track = filter_func(clean_backing_track)
        
        #sum the targets (excluding the everything else target)
        stacked_targets = torch.stack(targets_list, dim=0)
        
        #append the targets to backing track with an SnR
        min_length = np.min([processed_backing_track.shape[1], stacked_targets.shape[1]]) 
        stacked_targets = torch.narrow(stacked_targets, 1, 0, min_length)
        processed_backing_track = torch.narrow(processed_backing_track, 1, 0, min_length)
        
        #processed_audio_mix = (put the snr function from marius)
        
        #convolve the full mix with a room+mic ir
        processed_audio_mix, ir_info = apply_ir(self.room_and_mic_irs, processed_audio_mix) 
            
        #find the shortest audio length on dimension 1 and crop all to be at that length. Since all stems are the same size, we just use audio_sources[0]
        #min_length = np.min([targets_list[0].shape[1], convolved_backing_track.shape[1]])

        #should do another round of narrowing after making the convolution with the ir, so that all stems and outputs etc are the same length
        min_length = np.min([processed_audio_mix.shape[1], stacked_targets.shape[1], targets_list[0].shape[1]])
        for i in range(0, len(targets_list)):
            targets_list[i] = torch.narrow(targets_list[i], 1, 0, min_length)
            
        everything_else = torch.narrow(processed_backing_track, 1, 0, min_length) #should be attenuated relative to the SnR we did 
         
        #replace the pre-computed audio_mix with: ir(everything_else) + sum(everything in sources list)
        #everything_else = torch.tensor(apply_ir(self.irs, clean_backing_track), dtype=torch.float)

        targets_list.append(everything_else) #everything else added after targets_list is used to calc the mix
        covered_targets.append('everything_else')

        #now targets_list has each of the targets, and one entry for everything else
        stacked_targets = torch.stack(targets_list, dim=0)

        # and we can write as:
        # torchaudio.save('{}.wav'.format(self.targets[0]), sources_list[0], self.sample_rate). etc

        #convolve as follows: audio mix = Room * ( Speaker * everything_else + target instrument )

        #Adding noise as a target will be considered in the future, which would require stacked_audio_sources to include the noise used
        # to make the mix, and of course would require us to append the noise targets to what is in the conf file.
        
        return stacked_targets, covered_targets, processed_audio_mix, ir_info
        
    def generate_track(self, track_id):
        #unlike the dataloader, here index will refer to the number of the song.
        audio_sources = {}
       
        # load sources
        for source in self.sources:
            # we load full tracks
            audio, _ = sf.read(
            Path(self.tracks[track_id]["path"] / source).with_suffix(self.suffix),
            always_2d=True)

            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float)
            audio_sources[source] = audio

        audio_mix = torch.stack(list(audio_sources.values())).sum(0)

        if self.targets:
            covered_targets = [] # meaning, targets that are already processed. String array
            targets_list = []   
            for target in self.targets: # if target is an instrument, remove from audio_sources and keep track of it
                if target in audio_sources: 
                    covered_targets.append(target)
                    targets_list.append(audio_sources[target])

            #remove the targets from audio_sources. So that audio_sources has the backing track stems, and targets_list has the target stems
            for target in covered_targets:
                audio_sources.pop(target)

            # sum the targets that weren't covered.
            clean_backing_track = torch.stack(list(audio_sources.values())).sum(0)


        return audio_mix, clean_backing_track, audio_sources, targets_list, covered_targets 

    
    def generate_and_save_all(self):
        # iterate over all tracks and write them in the proper directory
        #inputs = {'full_mix': degraded_mix, ' }
        #          outputs = {}}
        
        outdir = self.output_train_data if self.split == 'train' else self.output_test_data

        for track_id in range(0, len(self.tracks)):
            songname = os.path.basename(os.path.normpath(self.tracks[track_id]["path"]))
            audio_mix, clean_backing_track, audio_sources, targets_list, covered_targets = self.generate_track(track_id)

            target_str = '-'.join(covered_targets[0:-1]) #string to represent all chosen targets for path calculation, not including the 'everything_else' at the end. Although this might be an overkill since effectively we'll only be separating one source anyway

            os.makedirs(os.path.join(outdir, target_str, songname), exist_ok=True)

            #params_file should be saved outside all the song folders.
            params_df = pd.DataFrame(columns=[]) #initialized as none since we want to base it on the outputs from the room parameters.

            #Create all relevant room variants for this song
            #TODO: check load the metadatafile if exists, and check if a particular variant exists before creating it.
            for i in np.arange(0, self.variants):
                print('{} - {}'.format(songname, i))
                r = self.room_factory.create_room()
                #pdb.set_trace()    
                os.makedirs(os.path.join(outdir, self.targets[0], songname, str(i)), exist_ok=True)
                r.add_instrument_track(targets_list[0])
                r.add_backing_track(torch.stack(list(audio_sources.values())).sum(0))
                
                #read degraded backing track as if there wasn't an instrument playing
                r.toggle_mute_backing_track()
                degraded_backing_track = normalize_and_convert(r.read_mic_output())

                #read degraded full mix
                r.toggle_mute_instrument_track()
                degraded_audio_mix = normalize_and_convert(r.read_mic_output())

                #read degraded instrument mix as if there were no backing track
                r.toggle_mute_backing_track()
                degraded_instrument_track = normalize_and_convert(r.read_mic_output())
                
                #pdb.set_trace()
                coordinates = r.get_coordinates()
                other_params = r.get_other_parameters()
                dimensions = r.get_dimensions()

                joint_dict = {}
                joint_dict.update(coordinates)
                joint_dict.update(other_params)
                joint_dict.update(dimensions)
                joint_dict.update({'songname': songname, 
                    'variant': i
                    })
                #Merge them, and add the songname and variant to them
                if params_df.empty:
                    #initialze the columns of params df
                    params_df = pd.DataFrame(columns = list(joint_dict.keys()))
                
                row_series = pd.Series(joint_dict)
                params_df = params_df.append(joint_dict, ignore_index=True)
                
                #appending to the last line
                #params_df.loc[len(params_df.index)] = [ir_info, songname]

                #pdb.set_trace()
                inputs = {
                  'degraded_audio_mix': degraded_audio_mix, 
                  'clean_backing_track': clean_backing_track
                 }

                #change this to use the degraded outputs
                outputs = {key: value for (key, value) in zip(covered_targets, targets_list)}
                outputs['degraded_backing_track'] = degraded_backing_track
                outputs['degraded_instrument_track'] = degraded_instrument_track
           
                # should write to folder target_str instead of self.targets[0], but currently there
                # is a little bug.
                for key, val in inputs.items():
                    torchaudio.save(os.path.join(outdir, self.targets[0], songname, str(i), '{}.wav'.format(key)), val, self.sample_rate)

                for key, val in outputs.items():
                    torchaudio.save(os.path.join(outdir, self.targets[0], songname, str(i), '{}.wav'.format(key)), val, self.sample_rate)

                #columns.extend(['songname', 'variant'])
                #params_df = pd.DataFrame(columns=columns)
                
            #keep writing in every forloop because we want to see intermediary results anyway
            params_df.to_csv(os.path.join(outdir, self.targets[0], 'room_params.csv'))

        #for track in self.tracks:
            # check the targets, and create a dataset for them
            # folder name should be x and y where x and y are the targets
            # call generate tracks on each.
            #torchaudio.save('{}.wav'.format(self.targets[0]), sources_list[0], self.sample_rate). etc
            #torchaudio.save(filepath:'everythingelse.wav', src: stacked_audio_sources[0] , sample_rate: self.sample_rate)
        return

    def get_irs(self, group): 
        """ Loads the impulse responses based on split and group. Currently there is nothing random about the ir selection """
        irs_df = pd.read_csv(self.ir_paths['irs_metadata'])
        irs_df = irs_df.fillna('')
        
        #get the irs marked relevant to the current split
        relevant_irs_df = irs_df[(irs_df['split'] == self.split & irs_df['group'] == group)]

        for index, ir_row in relevant_irs_df.iterrows():
            #add exception handling here, in case ir_paths doesn't have ir_row['Dataset']
            local_root =  self.ir_paths[ir_row['Dataset']]
            filepath = Path(local_root, ir_row['relative_path'], ir_row['Filename'])
            yield (filepath)


    def get_tracks(self):
        """Loads input and output tracks"""
        if self.split == 'train':
            p = Path(self.clean_train_data)
        else:
            p = Path(self.clean_test_data)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
               # if self.subset and track_path.stem not in self.subset:
                    # skip this track
               #     continue

                source_paths = [track_path / (s + self.suffix) for s in self.sources]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track due to non-existing source", track_path)
                    continue

                # get metadata
                infos = list(map(sf.info, source_paths))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print("Exclude track due to different sample rate ", track_path)
                    continue

                yield ({"path": track_path, "min_duration": None})


if __name__ == "__main__":
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    parser = argparse.ArgumentParser()
        
    with open("dataprep.yml") as f:
        def_conf = yaml.safe_load(f)
        parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    
    ir_paths = {'irs_metadata': Path(arg_dic['ir_paths']['irs_metadata']), 
               arg_dic['ir_paths']['irs_1']: Path(arg_dic['ir_paths']['irs_1_dir']), arg_dic['ir_paths']['irs_2']: Path(arg_dic['ir_paths']['irs_2_dir'])}
               
    #use partial to attach the room factory to the config parameters parameters
    #attached_room_factory = partial(room_factory, config)

    #for now we'll just paste the room factory parameters here.
    shoebox_units = [[2, 3], [2, 2]]
    room_units = [[[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]],
             [[0, 0], [0, 3], [5, 3], [5, 2.5], [3, 0]]]
    room_units_maxes = (3, 3) # range of max x and y for which we can place a shoebox in both unit shapes above
    room_fixed_1 = [[[0, 0], [0, 4], [6, 4], [6, 2], [3.5, 2], [3.5, 0]]]
    room_mult_factor = (1, 10)
    room_fixed_maxes = (3.5, 4)
    room_heights = [3, 3.5, 4]
    mic_placement_heights = [1, 1.5, 2]
    source_placement_heights = [0.5, 1, 1.5, 2]
    snr_options = []
    placement_margins = 0.25
    circle_radii = (0.75, 2.25)
    intrapoint_distance_min = 0.5
    intrapoint_distance_max = 2.5
    materials = ['hard_surface', 'carpet_cotton', 'brickwork', 'plasterboard', 'reverb_chamber', 'marble_floor', 'reverb_chamber']
    max_order = (2, 11)

    rf = RoomFactory(shoebox_units = shoebox_units,
            room_units = room_units,
            room_units_maxes = room_units_maxes,
            room_fixed = room_fixed_1,
            room_fixed_maxes = room_fixed_maxes,
            room_mult_factor = room_mult_factor,
            room_heights = room_heights,
            mic_placement_heights = mic_placement_heights,
            source_placement_heights = source_placement_heights,
            #snr_options = snr_options,
            placement_margins = placement_margins,
            circle_radii = circle_radii,
            intrapoint_distance_min = intrapoint_distance_min,
            intrapoint_distance_max = intrapoint_distance_max, 
            materials = materials,
            max_order = max_order)

    gen = MUSDB18LeakageDataGenerator(
            clean_train_data=arg_dic['data']['clean_train_dir'],
            clean_test_data=arg_dic['data']['clean_test_dir'],
            output_train_data=arg_dic['data']['out_train_dir'],
            output_test_data=arg_dic['data']['out_test_dir'],
            ir_paths=ir_paths, 
            sources=arg_dic['data']['sources'],
            targets=arg_dic['data']['targets'],
            room_factory = rf
            )
    gen.generate_and_save_all()

