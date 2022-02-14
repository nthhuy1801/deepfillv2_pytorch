# -------------------------
import os
import sys
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)
# -------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from utils.create_mask import *

class InpaintDataset(Dataset):
    def __init__(self, opt):
        super(InpaintDataset, self).__init__()
        self.opt = opt
        
        # Read folder, return full path
        self.img_list = []
        self.mask_list = opt.mask_dir
        for root, dirs, files in os.walk(opt.baseroot):
            for filepaths in files:
                self.img_list.append(os.path.join(root, filepaths))
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # Read image
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.img_size, self.opt.img_size))

        # Read mask
        mask = cv2.imread(self.mask_list[index])
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.opt.img_size, self.opt.img_size))

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).contiguous()
        return img, mask


if __name__=='__main__':
    from attrdict import AttrDict
    args = {
        'baseroot':'/home/huynth/deepfillv2_thesis/data/place2',
        'mask_dir':'/home/huynth/deepfillv2_thesis/data/mask',
        'img_size': 256
    }
    args = AttrDict(args)
    data = InpaintDataset(args)
    img, mask = data.__getitem__(1)
    print(mask.shape)
