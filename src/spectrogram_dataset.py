import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, labels, segment_paths, transform=None):
        self.labels = labels
        self.segment_paths = segment_paths
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec_path = self.segment_paths[idx]
        spectrogram = np.load(spec_path)
        spectrogram = torch.tensor(spectrogram).unsqueeze(0).float()

        if self.transform:
            spectrogram = self.transform(spectrogram)

        label = torch.tensor(self.labels[idx])
        track_dirname = os.path.dirname(spec_path)
        track_id = os.path.basename(track_dirname)
        return spectrogram, label, track_id