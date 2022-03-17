# -------------------------
import os
from tkinter import ALL
import glob
# -------------------------

import torch
from torch.utils.data import Dataset
from utils.create_mask import *

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']
class InpaintDataset(Dataset):
    def __init__(self, opt):
        super(InpaintDataset, self).__init__()
        self.opt = opt
        assert opt.mask_type in ALLMASKTYPES
        
        # Read folder, return full path
        self.img_list = []
        # for root, dirs, files in os.walk(opt.baseroot):
        #     for filepaths in files:
        #         self.img_list.append(os.path.join(root, filepaths))

        # # Nếu dataset là Places2"
        for img in glob.glob(os.path.join(opt.baseroot, "**/**/**/extra_train_*.jpg")):
            self.img_list.append(img)
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # Read image
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.image_size, self.opt.image_size))

        # Read mask
        # mask = cv2.imread(self.mask_list[index])
        # mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = cv2.resize(mask, (self.opt.img_size, self.opt.img_size))

        if self.opt.mask_type == 'single_bbox':
            mask = bbox2mask(shape=self.opt.image_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        elif self.opt.mask_type == 'bbox':
            mask = bbox2mask(shape = self.opt.image_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        elif self.opt.mask_type == 'free_form':
            mask = create_ff_mask(shape = self.opt.image_size, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        return img, mask


if __name__=='__main__':
    from attrdict import AttrDict
    args = {
        'baseroot':'/home/huynth/deepfillv2_thesis/data/place2',
        'mask_dir':'/home/huynth/deepfillv2_thesis/data/mask',
        'image_size': 256,
        'mask_type' : 'free_form'
    }
    args = AttrDict(args)
    data = InpaintDataset(args)
    img, mask = data.__getitem__(1)
    print(mask.shape)
