import torch
import torch.nn as nn
import torchvision
from module import *


def weights_init(net, init_type="kaiming", init_gain=0.02):
    def init_function(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("Initialization method [%s] is not implemented." % init_type)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    # Apply weight initialize function
    net.apply(init_function)


############### Generator #################
# Input: masked image + mask
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.coarse = nn.Sequential(
            # Encoder
            GatedCon2d(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type=opt.pad_type, activation=opt.activation, norm="none"),
            GatedCon2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type=opt.pad_type, activation = opt.activation, norm = opt.norm),
            
        )