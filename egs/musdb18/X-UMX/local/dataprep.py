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

#additions for the ir convolutions
import pandas as pd
import librosa
import numpy as np

'''
    Folder Structure:
        >>> #train/1/vocals.wav ---------|
        >>> #train/1/drums.wav ----------+--> input (mix), output[target]
        >>> #train/1/bass.wav -----------|
        >>> #train/1/other.wav ---------/
'''

parser = argparse.ArgumentParser()

def apply_ir(ir_paths, x):
    #maybe here I should make some choice based on a better random, because this will have the same path everytime
    # x is from sf.read and not librosa.load, but by the time it gets passed to this function is should have been already transposed and put in torch.audio 

    random_index = random.randint(0, len(ir_paths))
    ir_path = ir_paths[random_index]

    while(not ir_path.suffixes[-1] == '.wav'): #this could also be causing slowness
        ir_path = ir_paths[random.randint(0, len(ir_paths))]

    x_ir, fs = librosa.load(ir_path, sr=44100)

    x_ir = torch.tensor(x_ir, dtype=torch.float)

    #convert x_ir into stereo
    x_ir = torch.vstack([x_ir, x_ir])
    x = x.reshape(1, x.shape[0], x.shape[1])
    x_ir = x_ir.reshape((x_ir.shape[0], 1, x_ir.shape[1]))

    conv_result = torch.nn.functional.conv1d(x, x_ir, groups=2)
    conv_result = conv_result.reshape(conv_result.shape[1], conv_result.shape[2])

    return conv_result, ir_path 

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
        split='train',
        random_track_mix=False,
        sample_rate=44100,
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
        self.split = split
        self.tracks = list(self.get_tracks())
        self.irs = list(self.get_irs())
        if not self.tracks:
            raise RuntimeError("No tracks found.")

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
            covered_targets = [] # meaning, targets that are already processed
            targets_list = []   
            for target in self.targets: # if target is an instrument, remove from audio_sources and keep track of it
                if target in audio_sources: 
                    covered_targets.append(target)
                    targets_list.append(audio_sources[target])

            for target in covered_targets:
                audio_sources.pop(target)

            # sum the targets that weren't covered.
            clean_backing_track = torch.stack(list(audio_sources.values())).sum(0)

            convolved_backing_track, ir_info = apply_ir(self.irs, clean_backing_track)

            #find the shortest audio length on dimension 1 and crop all to be at that length. Since all stems are the same size, we just use audio_sources[0]
            min_length = np.min([targets_list[0].shape[1], convolved_backing_track.shape[1]])

            for i in range(0, len(targets_list)):
                targets_list[i] = torch.narrow(targets_list[i], 1, 0, min_length)
            everything_else = torch.narrow(convolved_backing_track, 1, 0, min_length)

            #replace the pre-computed audio_mix with: ir(everything_else) + sum(everything in sources list)
            #everything_else = torch.tensor(apply_ir(self.irs, clean_backing_track), dtype=torch.float)

            #targets_list is [[]] and everything_else is [], which is why we reshape everything_else below
            audio_mix = torch.stack(targets_list + [everything_else]).sum(0)

            targets_list.append(everything_else) #everything else added after targets_list is used to calc the mix
            covered_targets.append('everything_else')

            #now targets_list has each of the targets, and one entry for everything else
            stacked_targets = torch.stack(targets_list, dim=0)

            # and we can write as:
            # torchaudio.save('{}.wav'.format(self.targets[0]), sources_list[0], self.sample_rate). etc

            #convolve as follows: audio mix = Room * ( Speaker * everything_else + target instrument )

            #Adding noise as a target will be considered in the future, which would require stacked_audio_sources to include the noise used
            # to make the mix, and of course would require us to append the noise targets to what is in the conf file.

        return audio_mix, clean_backing_track, stacked_targets, covered_targets, ir_info 

    
    def generate_and_save_all(self):
        # iterate over all tracks and write them in the proper directory
        #inputs = {'full_mix': audio_mix, }
        #          outputs = {}}

        ir_metadata_df = pd.DataFrame(columns=['ir_path', 'songname'])
        outdir = self.output_train_data if self.split == 'train' else self.output_test_data

        for track_id in range(0, len(self.tracks)):
            songname = os.path.basename(os.path.normpath(self.tracks[track_id]["path"]))

            audio_mix, clean_backing_track, stacked_targets, covered_targets, ir_info= self.generate_track(track_id)

            target_str = '-'.join(covered_targets[0:-1]) #string to represent all chosen targets for path calculation, not including the 'everything_else' at the end
            os.makedirs(os.path.join(outdir, target_str, songname), exist_ok=True)

            ir_metadata_df.loc[len(ir_metadata_df.index)] = [ir_info, songname]

            inputs = {
                  'audio_mix': audio_mix, 
                  'clean_backing_track': clean_backing_track
                 }

            outputs = {key: value for (key, value) in zip(covered_targets, stacked_targets)}

            for key, val in inputs.items():
                torchaudio.save(os.path.join(outdir, target_str, songname, '{}.wav'.format(key)), val, self.sample_rate)

            for key, val in outputs.items():
                torchaudio.save(os.path.join(outdir, target_str, songname, '{}.wav'.format(key)), val, self.sample_rate)

            
            #keep writing in every forloop because we want to see intermediary results anyway
            ir_metadata_df.to_csv(os.path.join(outdir, target_str, 'ir_metadata.csv'))

        #for track in self.tracks:
            # check the targets, and create a dataset for them
            # folder name should be x and y where x and y are the targets
            # call generate tracks on each.
            #torchaudio.save('{}.wav'.format(self.targets[0]), sources_list[0], self.sample_rate). etc
            #torchaudio.save(filepath:'everythingelse.wav', src: stacked_audio_sources[0] , sample_rate: self.sample_rate)
        return

    def get_irs(self):
        """ Loads the impulse responses. Currently there is nothing random about the ir selection """
        irs_df = pd.read_csv(self.ir_paths['irs_metadata'])
        irs_df = irs_df.fillna('')

        #get the irs marked relevant to the current split
        relevant_irs_df = irs_df[irs_df['split'] == self.split]

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
    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)
        parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    
    ir_paths = {'irs_metadata': Path(arg_dic['ir_paths']['irs_metadata']), 
               arg_dic['ir_paths']['irs_1']: Path(arg_dic['ir_paths']['irs_1_dir']), arg_dic['ir_paths']['irs_2']: Path(arg_dic['ir_paths']['irs_2_dir'])}

    gen = MUSDB18LeakageDataGenerator(
            clean_train_data=arg_dic['data']['clean_train_dir'],
            clean_test_data=arg_dic['data']['clean_test_dir'],
            output_train_data=arg_dic['data']['out_train_dir'],
            output_test_data=arg_dic['data']['out_test_dir'],
            ir_paths=ir_paths, 
            sources=arg_dic['data']['sources'],
            targets=arg_dic['data']['targets']
            )
    gen.generate_and_save_all()

