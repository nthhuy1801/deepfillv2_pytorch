import os
import torch
import torch.nn as nn
from options.test_options import TestOptions
from dataset.test_dataset import TestInpaintDataset

opt = TestOptions().parse()
print(opt)
def tester(opt):
    def load_model_generator(net, epoch, opt):
        pass