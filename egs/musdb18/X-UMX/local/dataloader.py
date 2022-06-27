from asteroid.data import Leakage_MUSDB18Dataset
import torch
from pathlib import Path

validation_tracks = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    dataset_kwargs = {
        "root": Path(args.root),
    }

    #if args.leakage_removal:
    #    dataset_kwargs['ir_paths'] = {'irs_metadata': Path(args.irs_metadata), args.irs_1: Path(args.irs_1_dir), args.irs_2: Path(args.irs_2_dir)}
    #    dataset_kwargs['leakage_removal'] = True

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    train_dataset = Leakage_MUSDB18Dataset(
        #root=args.root,
        split="train",
        inputs=args.inputs,
        outputs=args.outputs,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        **dataset_kwargs,
    )
    train_dataset = filtering_out_valid(train_dataset)

    valid_dataset = Leakage_MUSDB18Dataset(
        #root=args.root,
        split="train",
        subset=validation_tracks,
        inputs=args.inputs,
        outputs=args.outputs,
        segment=None,
        **dataset_kwargs,
    )

    return train_dataset, valid_dataset


def filtering_out_valid(input_dataset):
    """Filtering out validation tracks from input dataset.

    Return:
        input_dataset (w/o validation tracks)
    """
    input_dataset.tracks = [
        tmp
        for tmp in input_dataset.tracks
        if not (str(tmp["path"]).split("/")[-1] in validation_tracks)
    ]

    return input_dataset


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio

#does augment Snr also make sense to be added here

def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`"""
    gain = low + torch.rand(1) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])

    return audio

#here, we must add the augmentations we want, which are the noise additions. 
#def _augment_ir_conv(audio, ir_info='', purpose='train'):
#    """ Apply noise convolutions to the audio depending on the purpose. ir_info leads to a csv with the irs to be used for train vs test vs validation, and we filter for a relevant ir depending on the purpose"""

#    return audio
