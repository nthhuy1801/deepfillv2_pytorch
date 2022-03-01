import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import weights_init
from utils.utils import create_generator, create_discriminator, create_perceptualnet, save_sample_png
from dataset.train_dataset import InpaintDataset

def WGANTrainer(opt):
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
        

    # Build networks
    generator = create_generator(opt)
    discriminator = create_discriminator(opt)
    perceptualnet = create_perceptualnet()


    # Loss function
    L1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    
    # Set up Adam optimizers for G & D
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), 
                                    weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), 
                                    weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by 
        "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        model_name = 'deepfillv2_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if epoch%opt.checkpoint_interval==0:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch {}'.format(epoch))

    # Save discriminator
    def save_model_disc(net, epoch, opt):
        model_name = 'deepfillv2_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if epoch%opt.checkpoint_interval==0:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch'.format(epoch))
    
    # Load model for inferring
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'deepfillv2_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'deepfillv2_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('Successful loaded pretrained model')

    # Load model to GPU
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    perceptualnet = perceptualnet.cuda()

    # Define the train set
    trainset = InpaintDataset(opt)
    # Define dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, 
                        num_workers = opt.num_workers, pin_memory = True)

    # Initialize start time
    start_time = time.time()

    # Create Tensor cuda
    # Tensor = torch.cuda.FloatTensor

        # training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, (img, mask) in enumerate(dataloader):
            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), 
            # img (shape: [B, 3, H, W]) and put it to cuda
            img = img.cuda()
            mask = mask.cuda()

            # Train discriminator
            optimizer_d.zero_grad()

            # Generator ouput
            first_out, second_out = generator(img, mask)

            # Forward propagation
            first_out_whole_img = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_whole_img = img * (1 - mask) + second_out * mask      # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(second_out_whole_img.detach(), mask)
            # True samples
            true_scalar = discriminator(img, mask)

            # Overall Loss and optimize
            loss_D =  - torch.mean(true_scalar) + torch.mean(fake_scalar)
            loss_D.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()

            # Mask L1 Loss
            first_MaskL1Loss = L1_loss(first_out_whole_img, img)
            second_MaskL1Loss = L1_loss(second_out_whole_img, img)

            # GAN Loss
            fake_scalar = discriminator(second_out_whole_img, mask)
            GAN_Loss = -torch.mean(fake_scalar)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)                            # feature maps
            second_out_wholeimg_featuremaps = perceptualnet(second_out_whole_img)
            sec_percept_loss = L1_loss(second_out_wholeimg_featuremaps, img_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
                        opt.lambda_perceptual * sec_percept_loss + opt.lambda_gan * GAN_Loss
            loss.backward()
            optimizer_g.step()

            # Time execute
            time_left = time.time() - start_time
            start_time = time.time()

            # Print log
            print("\r[Epoch {}/{}] [Batch {}/{}] [First Mask L1 Loss: {:.4f}] [Second Mask L1 Loss: {:.4f}]".format((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(), second_MaskL1Loss.item()))
            print("\r[D Loss: {:.4f}] [G Loss: {:.4f}] [Perceptual Loss: {:.4f}] Time: {}s" .format(
                loss_D.item(), GAN_Loss.item(), sec_percept_loss.item(), time_left))
            print('-'*50)
            if (batch_idx + 1) % 1000 == 0:
                torch.save(generator.state_dict(), 'deepfillv2_G_epoch%d_batchsize%d_batchidx%d.pth' % (epoch, opt.batch_size, batch_idx))
                torch.save(discriminator.state_dict(), 'deepfillv2_D_epoch%d_batchsize%d_batchidx%d.pth' % (epoch, opt.batch_size, batch_idx))

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)
        # Save the model
        save_model_generator(generator, (epoch + 1), opt)
        save_model_disc(discriminator, (epoch + 1), opt)

        # Sample data every epoch
        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
            save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, 
                            name_list = name_list, pixel_max_cnt = 255)