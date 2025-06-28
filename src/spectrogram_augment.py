import torch
import numpy as np
import random

class SpectrogramAugment:
    def __init__(self, time_mask_param=20, freq_mask_param=10, noise_level=0.01, shift_max=10,
                 time_mask_prob=0.5, freq_mask_prob=0.5, shift_prob=0.5, noise_prob=0.5):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_level = noise_level
        self.shift_max = shift_max
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.shift_prob = shift_prob
        self.noise_prob = noise_prob

    def __call__(self, spec):
        if isinstance(spec, np.ndarray):
            spec = torch.tensor(spec).unsqueeze(0)

        # Frequency masking
        if random.random() < self.freq_mask_prob:
            freq = spec.shape[1]
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, freq - f))
            spec[:, f0:f0+f, :] = spec.min()#-80

        # Time masking
        if random.random() < self.time_mask_prob:
            time = spec.shape[2]
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, time - t))
            spec[:, :, t0:t0+t] = spec.min()#-80

        # Time shifting
        if random.random() < self.shift_prob:
            shift = random.randint(-self.shift_max, self.shift_max)
            spec = torch.roll(spec, shifts=shift, dims=2)

        # Gaussian noise
        if random.random() < self.noise_prob:
            noise = torch.randn_like(spec) * self.noise_level
            spec = spec + noise

        return spec