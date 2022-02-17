import os
import time
import datetime
import pickle as pkl
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.model import weights_init
from utils.utils import create_generator, create_discriminator, create_perceptualnet, save_sample_png
from dataset.dataset import InpaintDataset

class WGANTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, opt):
        self.opt = opt

        # self.lr_g = opt.lr_g
        # self.lr_d = opt.lr_d
        # self.b1 = opt.b1
        # self.b2 = opt.b2
        # cudnn.benchmark = opt.cudnn.benchmark

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

        self.resume = os.path.join(self.save_folder, 'model.pth')
        self.history = os.path.join(self.save_folder, 'history.pkl')
        

    # Learning rate decrease
    def adjust_learning_rate(self, lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # Save model
    def save(self, epoch):
        # model_name = os.path.join(self.save_folder, f'deepfillv2_wgan_epoch_{epoch}_{opt.batch_size}.pth')
        """[function to save model]
        Args:
            epoch ([int]): [number of epochs]
        """
        checkpoint = {}
        checkpoint['iters'] = self.iters
        checkpoint['epoch'] = epoch
        checkpoint['G'] = self.G.state_dict()
        checkpoint['D'] = self.D.state_dict()
        datas = {}
        datas["G_loss"] = self.G_losses
        datas["D_loss"] = self.D_losses
        resume = os.path.join(self.save_folder, "model_{}.pth".format(str(epoch)))
        torch.save(checkpoint, resume)  
        # torch.save(checkpoint, self.resume)
        history = os.path.join(self.save_folder, "history_{}.pkl".format(str(epoch)))
        pkl.dump(datas, open(history, 'wb')) 
        
        self.G_losses = []
        self.D_losses = []  
        print("* {} saved.\n* {} saved.".format(resume, history))       
        print('Model saved at epoch %d' % (epoch))

    def load(self):
        """[fucntion to load model]
        """
        checkpoint = torch.load(self.resume)
        self.start_epoch = checkpoint['epoch']
        self.iters = checkpoint['iters']
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        if os.path.exists(self.history):
            datas = pkl.load(open(self.history, 'rb'))
            self.G_losses = datas["G_loss"]
            self.D_losses = datas["D_loss"]
        else:
            self.G_losses = []
            self.D_losses = []
        print("* {} loaded.\n* {} loaded.".format(self.resume, self.history))

    def train(self, opt):
        if os.path.exists(self.resume):
            self.load()
        else:
            self.start_epoch = 0
            self.iters = 0
            self.G_losses = []
            self.D_losses = []
        self.G.cuda()
        self.D.cuda()

        # Set up Adam optimizers for G & D
        optimizer_g = torch.optim.Adam(self.G.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), 
                                        weight_decay = opt.weight_decay)
        optimizer_d = torch.optim.Adam(self.D.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), 
                                        weight_decay = opt.weight_decay)

        # Loss functions
        L1Loss = nn.L1Loss()
        # Define the train set
        trainset = InpaintDataset(opt)
        # Define dataloader
        dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, 
                            num_workers = opt.num_workers, pin_memory = True)

        # Initialize start time
        start_time = time.time()

        # training loop
        for epoch in range(opt.epochs):
            for batch_idx, (img, mask) in dataloader:
                # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), 
                # img (shape: [B, 3, H, W]) and put it to cuda
                img = img.cuda()
                mask = mask.cuda()

                # Train discriminator
                optimizer_d.zero_grad()

                # Generator ouput
                first_out, second_out = self.G(img, mask)

                # Forward propagation
                first_out_whole_img = img * (1 - mask) + first_out * mask        # in range [0, 1]
                second_out_whole_img = img * (1 - mask) + second_out * mask      # in range [0, 1]

                # Fake samples
                fake_scalar = self.D(second_out_whole_img.detach(), mask)
                # True samples
                true_scalar = self.D(img, mask)

                # Overall Loss and optimize
                loss_D = torch.mean(fake_scalar) - torch.mean(true_scalar)
                loss_D.backward()
                optimizer_d.step()

                # Train generator
                optimizer_g.zero_grad()

                # Mask L1 Loss
                first_MaskL1Loss = L1Loss(first_out_whole_img, img)
                second_MaskL1Loss = L1Loss(second_out_whole_img, img)

                # GAN Loss
                fake_scalar = self.D(second_out_whole_img, mask)
                GAN_Loss = - torch.mean(fake_scalar)

                # Get the deep semantic feature maps, and compute Perceptual Loss
                img_featuremaps = self.percep_net(img)                            # feature maps
                second_out_wholeimg_featuremaps = self.percep_net(second_out_whole_img)
                second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

                # Compute losses
                loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
                    opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
                loss.backward()
                optimizer_g.step()

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + batch_idx
                batches_left = opt.epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - start_time))
                start_time = time.time()

                # Print log
                print("\r[Epoch {}/{}] [Batch {}/{}] [First Mask L1 Loss: {:.4f}] [Second Mask L1 Loss: {:.4f}]".format(
                    ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(), second_MaskL1Loss.item())))
                print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                    (loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item(), time_left))
                print('-'*30)
            # Learning rate decrease
        self.adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        self.adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        self.save((epoch + 1))

        ### Sample data every epoch
        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
            save_sample_png(sample_folder = self.sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, 
                            name_list = name_list, pixel_max_cnt = 255)