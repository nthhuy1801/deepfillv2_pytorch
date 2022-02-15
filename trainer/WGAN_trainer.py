import os
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils import create_generator, create_discriminator, create_perceptualnet

class WGANTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, opt):
        self.opt = opt

        cudnn.benchmark = opt.cudnn.benchmark

        # configurations
        self.save_folder = opt.save_path
        self.sample_folder = opt.sample_path
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if not os.path.exists(self.sample_folder):
            os.makedirs(self.sample_folder)

        # Build network
        self.G = create_generator(self.opt).cuda()
        self.D = create_discriminator(self.opt).cuda()
        self.percep_net = create_perceptualnet().cuda()

        

    # Learning rate decrease
    def adjust_learning_rate(self, lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # Save model
    def save_model(self, net, epoch, opt):
        model_name = os.path.join(self.save_folder, f'deepfillv2_wgan_epoch_{epoch}_{opt.batch_size}.pth')
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_name)
            print('Model saved at epoch %d' % (epoch))

    def load_model(self, epoch):


