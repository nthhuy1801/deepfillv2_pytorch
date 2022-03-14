import cv2
import os
from cv2 import sort
import torch
import numpy as np
from torch.utils.data import Dataset

class TestInpaintDataset(Dataset):
    def __init__(self, opt):
        super(TestInpaintDataset, self).__init__()
        self.opt = opt
        # Image list
        self.img_list = []
        for root, dirs, files in os.walk(opt.baseroot):
            for filepaths in files:
                self.img_list.append(os.path.join(root, filepaths))
        self.img_list = sorted(self.img_list)
        # Mask list
        self.mask_list = []
        for root, dirs, files in os.walk(opt.maskroot):
            for filepaths in files:
                self.mask_list.append(os.path.join(root, filepaths))
        self.mask_list = sorted(self.mask_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # Read mask
        img_name = self.img_list[index]
        img_path = os.path.join(self.opt.baseroot, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.image_size, self.opt.image_size))
        # Read mask
        mask = cv2.imread(self.mask_list[index])[:, :, 0]
        mask = cv2.resize(mask, (self.opt.image_size, self.opt.image_size))
        # To tensor
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        return img, mask
