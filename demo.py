import cv2
import os
import torch
import numpy as np
from glob import glob
from models.model import Generator 
from torchvision.transforms import ToTensor


def postProcess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def demo(opt):
    # Load images
    img_list = []
    for ext in ['*.jpg', '*.png']:
        img_list.extend(glob(os.path.join(opt.test_dir, ext)))
    img_list.sort()

    # Load model
    model = Generator(opt)
    model.load_state_dict(torch.load(opt.pretrained, map_location='cpu'))
    model.eval()
    print("Loading pretrained model successful !!!")

    for fn in img_list:
        filename = os.path.basename(fn).split('.')[0]
        orig_img = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (256,256))
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
        
