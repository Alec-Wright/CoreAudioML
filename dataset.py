import numpy as np
from scipy.io import wavfile
import torch
import math
import warnings
import os


# Function converting np read audio to range of -1 to +1
def audio_converter(audio):
    if audio.dtype == 'int16':
        return audio.astype(np.float32, order='C') / 32768.0
    else:
        print('unimplemented audio data type conversion...')


# Splits audio, each split marker determines the fraction of the total audio in that split, i.e [0.75, 0.25] will put
# 75% in the first split and 25% in the second
def audio_splitter(audio, split_markers):
    assert sum(split_markers) <= 1.0
    if sum(split_markers) < 0.999:
        warnings.warn("sum of split markers is less than 1, so not all audio will be included in dataset")
    start = 0
    slices = []
    # convert split markers to samples
    split_bounds = [int(x * audio.shape[0]) for x in split_markers]
    for n in split_bounds:
        end = start + n
        slices.append(audio[start:end])
        start = end
    return slices


# converts numpy audio into frames, and creates a torch tensor from them, frame_len = 0 just converts to a torch tensor
def framify(audio, frame_len):
    # If audio is mono, add a dummy dimension, so the same operations can be applied to mono/multichannel audio
    audio = np.expand_dims(audio, 1) if len(audio.shape) == 1 else audio
    # Calculate the number of segments the training data will be split into in frame_len is not 0
    seg_num = math.floor(audio.shape[0] / frame_len) if frame_len else 1
    # If no frame_len is provided, set frame_len to be equal to length of the input audio
    frame_len = audio.shape[0] if not frame_len else frame_len
    # Find the number of channels
    channels = audio.shape[1]
    # Initialise tensor matrices
    dataset = torch.empty((frame_len, seg_num, channels))
    # Load the audio for the training set
    for i in range(seg_num):
        dataset[:, i, :] = torch.from_numpy(audio[i * frame_len:(i + 1) * frame_len, :])
    return dataset


"""This is the main DataSet class, it can hold any number of subsets, which could be, e.g, the training/test/val sets.
The subsets are created by the create_subset method and stored in the DataSet.subsets dictionary. 

datadir: is the default location where the DataSet instance will look when told to load an audio file
extensions: is the default ends of the paired data files, when using paired data, so by default when loading the file
 'wicked_guitar', the load_file method with look for 'wicked_guitar-input.wav' and 'wicked_guitar-target.wav'
 to disable this behaviour, enter extensions = '', or None, or anything that evaluates to false in python """


class DataSet:
    def __init__(self, data_dir='../Dataset/', extensions=('input', 'target')):
        self.extensions = extensions if extensions else ['']
        self.subsets = {}
        assert type(data_dir) == str, "data_dir should be string,not %r" % {type(data_dir)}
        self.data_dir = data_dir

    # add a subset called 'name', desired 'frame_len' is given in seconds, or 0 for just one long frame
    def create_subset(self, name, frame_len=0):
        assert type(name) == str, "data subset name must be a string, not %r" %{type(name)}
        assert not (name in self.subsets), "subset %r already exists" %name
        self.subsets[name] = SubSet(frame_len)

    # load a file of 'filename' into existing subset/s 'set_names', split fractionally as specified by 'splits',
    # if 'cond_val' is provided the conditioning value will be saved along with the frames of the loaded data
    def load_file(self, filename, set_names='train', splits=None, cond_val=None):
        # Assertions and checks
        if type(set_names) == str:
            set_names = [set_names]
        assert len(set_names) == 1 or len(set_names) == len(splits), "number of subset names must equal number of " \
                                                                     "split markers"
        assert [self.subsets.get(each) for each in set_names], "set_names contains subsets that don't exist yet"


        # Load each of the 'extensions'
        for i, ext in enumerate(self.extensions):
            try:
                file_loc = os.path.join(self.data_dir, filename + '-' + ext)
                file_loc = file_loc + '.wav' if not file_loc.endswith('.wav') else file_loc
                np_data = wavfile.read(file_loc)
            except FileNotFoundError:
                file_loc = os.path.join(self.data_dir, file_loc + ext)
                file_loc = file_loc + '.wav' if not file_loc.endswith('.wav') else file_loc
                np_data = wavfile.read(file_loc)
            except FileNotFoundError:
                print(["File Not Found At: " + self.data_dir + filename])
                return
            raw_audio = audio_converter(np_data[1])
            # Split the audio if the set_names were provided
            if len(set_names) > 1:
                raw_audio = audio_splitter(raw_audio, splits)
                for n, sets in enumerate(set_names):
                    self.subsets[set_names[n]].add_data(np_data[0], raw_audio[n], ext, cond_val)
            elif len(set_names) == 1:
                self.subsets[set_names[0]].add_data(np_data[0], raw_audio, ext, cond_val)


# The SubSet class holds a subset of data,
# frame_len sets the length of audio per frame (in s), if set to 0 a single frame is used instead
class SubSet:
    def __init__(self, frame_len):
        self.data = {}
        self.cond_data = {}
        self.frame_len = frame_len
        self.conditioning = None
        self.fs = None

    # Add 'audio' data, in the data dictionary at the key 'ext', if cond_val is provided save the cond_val of each frame
    def add_data(self, fs, audio, ext, cond_val):
        if not self.fs:
            self.fs = fs
        assert self.fs == fs, "data with different sample rate provided to subset"
        # if no 'ext' is provided, all the subsets data will be stored at the 'data' key of the 'data' dict
        ext = 'data' if not ext else ext
        # Frame the data and optionally create a tensor of the conditioning values of each frame
        framed_data = framify(audio, self.frame_len)
        cond_data = cond_val * torch.ones(framed_data.shape[1]) if isinstance(cond_val, (float, int)) else None

        try:
            # Convert data from tuple to list and concatenate new data onto the data tensor
            data = list(self.data[ext])
            self.data[ext] = (torch.cat((data[0], framed_data), 1),)
            # If cond_val is provided add it to the cond_val tensor, note all frames or no frames must have cond vals
            if isinstance(cond_val, (float, int)):
                assert torch.is_tensor(self.cond_data[ext][0]), 'cond val provided, but previous data has no cond val'
                c_data = list(self.cond_data[ext])
                self.cond_data[ext] = (torch.cat((c_data[0], cond_data), 0),)
        # If this is the first data to be loaded into the subset, create the data and cond_data tuples
        except KeyError:
            self.data[ext] = (framed_data,)
            self.cond_data[ext] = (cond_data,)
