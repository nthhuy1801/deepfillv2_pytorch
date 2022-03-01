import os
import sys
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import *
from torchvision.models import vgg16



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
            GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, 
                        pad_type=opt.pad_type, activation=opt.activation, norm="none"),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, 
                        pad_type=opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, 
                        pad_type=opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, 
                        pad_type=opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottle neck 
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 
                        pad_type=opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 
                        pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, 
                        dilation=2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, 
                        dilation=4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, 
                        dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, 
                        dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 
                        pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 
                        pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Decoder
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, 
                        pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, 
                        pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, 
                        pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # GatedConv2d(opt.latent_channels, opt.latent_channels//2, 3, 1, 1, 
            #             pad_type = opt.pad_type, activation = opt.activation, norm=opt.norm),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, 
                            pad_type = opt.pad_type, activation = 'Tanh', norm = opt.norm),
            # nn.Tanh()
        )
        # Refinement
        self.refine = nn.Sequential(
            # Encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, 
                        pad_type = opt.pad_type, activation = opt.activation, norm="none"),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm=opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, 
                            activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, 
                            activation = opt.activation, norm = opt.norm),
            # Decoder 
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, 
                            pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, 
                            pad_type = opt.pad_type, activation = 'Tanh', norm = 'none')                
            )
        # self.refine_atten1 = nn.Sequential(
        #     GatedConv2d(opt.in_channels, opt.latent_channels , 5, 1, 2, 
        #                     pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels, opt.latent_channels, 3, 2, 1,
        #                     pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
        #     GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 1, 1,
        #                     pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
        #     GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1,
        #                     pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1,
        #                     pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
        #     GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 
        #                     pad_type = opt.pad_type, activation = 'ReLU', norm = opt.norm)
        # )
        # self.refine_atten2 = nn.Sequential(
        #     GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 
        #                     pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 
        #                     pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
        # )
        # self.refine_combine = nn.Sequential(
        #     GatedConv2d(opt.latent_channels*8, opt.latent_channels*4, 3, 1, 1, 
        #                 pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 
        #                 pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels*2, 3, 1, 1, 
        #                 pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels*2, opt.latent_channels*2, 3, 1, 1, 
        #                 pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels, opt.latent_channels//2, 3, 1, 1, 
        #                 pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels//2, opt.out_channels, 3, 1, 1, 
        #                 pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
        #     nn.Tanh()
        # )
        # self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
        #                                              fuse=True)
    
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        # print(img.shape, mask.shape)
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat([first_masked_img, mask], dim=1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                           # out: [B, 3, H, W]
        # first_out = F.interpolate(first_out, (img.shape[2], img.shape[3]))
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        second_out = self.refine(second_in)
        # refine_attention = self.refine_atten1(second_in)
        # mask_s = F.interpolate(mask, (refine_attention.shape[2], refine_attention.shape[3]))
        # refine_attention = self.context_attention(refine_attention, refine_attention, mask_s)
        # refine_attention = self.refine_atten2(refine_attention)
        # second_out = torch.cat([refine_conv, refine_attention], dim=1)
        # second_out = self.refine_combine(second_out)
        # second_out = F.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out


#### Discriminator ###
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        # Down sample
        self.block1 = Conv2dLayer(opt.in_channels, opt.latent_channels, 7, 1, 3,
                    pad_type=opt.pad_type, activation=opt.activation, norm='none', sn=True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1,
                    pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1,
                    pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1,
                    pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1,
                    pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1,
                    pad_type=opt.pad_type, activation="none", norm="none", sn=True)

    def forward(self, img, mask):
        # Input: ground truth + mask 
        # Output: patch base region
        x = torch.cat([img, mask], dim=1) # In channel = 4
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x

# Perceptual network
# VGG-16 conv4_3 features
class PerceptualNetwork(nn.Module):
    def __init__(self):
        super(PerceptualNetwork, self).__init__()
        block = [
            vgg16(pretrained=True).features[:22].eval()
        ]
        for p in block[0]:
            p.requires_grad = False
        self.block = nn.ModuleList(block)
        self.transform = F.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x
        
if __name__ == "__main__":
    from attrdict import AttrDict
    from torchsummary import summary
    args = {
        'in_channels': 4,
        'latent_channels': 48,
        'out_channels': 3,
        'pad_type': 'zero',
        'activation': 'ELU',
        'norm': 'none'
    }
    args = AttrDict(args)
    # model = Generator(args)
    # print(summary(model, [(3,128,128), (1,128,128)]))
    model = Discriminator(args)
    print(summary(model, [(3,256,256), (1,256,256)]))