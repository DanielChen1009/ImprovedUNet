import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class DisasterDataset(Dataset):

    def __init__(self, image_dir, mask_dir, start, end, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.images.sort()
        self.images = self.images[start:end]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        post_path = os.path.join(self.image_dir, self.images[2*item])
        pre_path = os.path.join(self.image_dir, self.images[2*item+1])
        mask_path = os.path.join(self.mask_dir, self.images[2*item].replace(".png", "_target.png"))
        post = np.array(Image.open(post_path).convert("RGB"))
        pre = np.array(Image.open(pre_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(pre=pre, image=post, mask=mask)
            post = augmentations["image"]
            pre = augmentations["pre"]
            mask = augmentations["mask"]
        return pre, post, mask
