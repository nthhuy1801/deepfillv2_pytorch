import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from options.test_options import TestOptions
from utils.utils import create_generator
from PIL import Image

# Options
opt = TestOptions().parse()

def load_model(generator, epoch, opt):
    pre_dict = torch.load(opt.load_name, map_location='cpu')
    generator.load_state_dict(pre_dict)

    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

# Load moel generator
model = create_generator(opt).eval()
load_model(model, opt.epochs, opt)
print("Load pretrained model generator")


def predict(img, mask, opt=None):
    img = img.convert('RGB')
    img_raw = np.array(img)
    w_raw, h_raw = img.size
    h_t, w_t = h_raw // 8*8, w_raw // 8*8

    img = img.resize((w_t, h_t))
    img = np.array(img).transpose(2,0,1)

    mask_raw = np.array(mask)[..., None] > 0
    mask = mask.resize((w_t, h_t))

    mask = np.array(mask)
    mask = (torch.Tensor(mask) > 0).float()
    img = (torch.Tensor(img)).float()

    img = (img / 255 - 0.5) / 0.5
    img = img[None]
    mask = mask[None, None]

    with torch.no_grad():
        generated, _ = model(img , mask)
    
    generated = torch.clamp(generated, -1, 1)
    generated = (generated + 1) / 2 * 255
    generated = generated.cpu().nupy().astype(np.uint8)
    generated = generated[0].transpose((1, 2, 0))
    result = generated * mask_raw + img_raw * (1 - mask_raw)
    result = result.astype(np.unit8)

    # result = Image.fromarray(result).resize((w_raw, h_raw))
    # result = np.array(result)
    # result = Image.fromarray(result.astype(np.uint8))
    # result.save(f'static/results/{name}')