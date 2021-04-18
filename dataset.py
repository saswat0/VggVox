import os

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal

def cmvnw(vec, win_size=301, variance_normalization=False):
    # Implementation from https://github.com/astorfi/speechpy
    """ This function is aimed to perform local cepstral mean and
    variance normalization on a sliding window. The code assumes that
    there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        win_size (int): The size of sliding window for local normalization.
            Default=301 which is around 3s if 100 Hz rate is
            considered(== 10ms frame stide)
        variance_normalization (bool): If the variance normilization should
            be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    # Get the shapes
    eps = 2**-30
    rows, cols = vec.shape

    # Windows size must be odd.
    assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert win_size % 2 == 1, "Windows size must be odd!"

    # Padding and initial definitions
    pad_size = int((win_size - 1) / 2)
    vec_pad = np.lib.pad(vec, ((pad_size, pad_size), (0, 0)), 'symmetric')
    mean_subtracted = np.zeros(np.shape(vec), dtype=np.float32)

    for i in range(rows):
        window = vec_pad[i:i + win_size, :]
        window_mean = np.mean(window, axis=0)
        mean_subtracted[i, :] = vec[i, :] - window_mean

    # Variance normalization
    if variance_normalization:

        # Initial definitions.
        variance_normalized = np.zeros(np.shape(vec), dtype=np.float32)
        vec_pad_variance = np.lib.pad(
            mean_subtracted, ((pad_size, pad_size), (0, 0)), 'symmetric')

        # Looping over all observations.
        for i in range(rows):
            window = vec_pad_variance[i:i + win_size, :]
            window_variance = np.std(window, axis=0)
            variance_normalized[i, :] \
            = mean_subtracted[i, :] / (window_variance + eps)
        output = variance_normalized
    else:
        output = mean_subtracted

    return output

class VoxLoader(Dataset):
    
    def __init__(self, path, train, transform=None):
        iden_split_path = os.path.join(path, 'iden_split.txt')
        split = pd.read_table(iden_split_path, sep=' ', header=None, names=['phase', 'path'])
        
        if train:
            phases = [1, 2]
        
        else:
            phases = [3]
            
        mask = split['phase'].isin(phases)
        self.dataset = split['path'][mask].reset_index(drop=True)
        self.path = path
        self.train = train
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        track_path = self.dataset[idx]
        audio_path = os.path.join(self.path, 'audio', track_path)

        rate, samples = wavfile.read(audio_path)
        label = int(track_path.split('/')[0].replace('id1', '')) - 1

        window = 'hamming'
        Tw = 25; Ts = 10
        
        Nw = int(rate * Tw * 1e-3)
        Ns = int(rate * (Tw - Ts) * 1e-3)

        nfft = 2 ** (Nw - 1).bit_length()
        pre_emphasis = 0.97
        
        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
        
        samples = signal.lfilter([1, -1], [1, -0.99], samples)
        dither = np.random.uniform(-1, 1, samples.shape)
        spow = np.std(samples)
        samples = samples + 1e-6 * spow * dither
        
        if self.train:
            segment_len = 3
            upper_bound = len(samples) - segment_len * rate
            start = np.random.randint(0, upper_bound)
            end = start + segment_len * rate
            samples = samples[start:end]
        
        _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft, 
                                        mode='magnitude', return_onesided=False)
        spec = cmvnw(spec, win_size=3 * rate)
        spec *= rate / 10
        
        if self.transform:
            spec = self.transform(spec)

        return label, spec