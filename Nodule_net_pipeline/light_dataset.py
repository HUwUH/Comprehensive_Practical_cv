import torch
from torch.utils.data import Dataset
import numpy as np
import nrrd
import os
import math

def pad2factor(image, factor=16, pad_value=0):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = []
    pad.append([0, d - depth])
    pad.append([0, h - height])
    pad.append([0, w - width])

    image = np.pad(image, pad, 'constant', constant_values=pad_value)
    return image

class InferenceReader(Dataset):
    def __init__(self, data_dir, filenames, cfg):
        self.data_dir = data_dir
        self.filenames = filenames  # list of pids
        self.cfg = cfg
        self.pad_value = cfg.get("pad_value", 170)
        self.factor = cfg.get("pad_factor", 16)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        pid = self.filenames[idx]
        img, _ = nrrd.read(os.path.join(self.data_dir, f'{pid}_clean.nrrd'))  # (D, H, W)

        original_shape = img.shape

        # pad
        img = pad2factor(img, factor=self.factor, pad_value=self.pad_value)

        # add channel dim
        img = img[np.newaxis, ...]  # (1, D, H, W)

        # normalize
        input_tensor = (img.astype(np.float32) - 128.) / 128.

        return torch.from_numpy(input_tensor).float(), pid, original_shape
