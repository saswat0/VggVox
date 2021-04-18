import torch
import numpy as np

class Normalize(object):
    """Normalise voice spectrogram (mean-variance)"""
    
    def __call__(self, spec):

        mu = spec.mean(axis=1).reshape(512, 1)
        sigma = spec.std(axis=1).reshape(512, 1)
        spec = (spec - mu) / sigma

        return spec

class ToTensor(object):
    
    def __call__(self, spec):
        F, T = spec.shape
        
        spec = spec.astype(np.float32) # F x T
        spec = torch.from_numpy(spec).unsqueeze(0)  # 1 x F x T
        
        return spec