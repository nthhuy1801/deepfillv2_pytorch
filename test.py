import os
import torch
import torch.nn as nn
from options.test_options import TestOptions
from dataset.test_dataset import TestInpaintDataset
from utils.utils import create_generator
from torch.utils.data import DataLoader
from metrics.metrics import *
from utils.utils import save_sample_png

opt = TestOptions().parse()
print(opt)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Inpain_Test(opt):
    def load_model(generator, epoch, opt):
        model_name = 'deepfillv2_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join('save_models', model_name)
        pre_dict = torch.load(model_name)
        generator.load_state_dict(pre_dict)

    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    # Load model for testing
    generator = create_generator(opt).eval().to(device)
    load_model(generator, opt.epochs, opt)
    print("Load pretrained model generator")


    testset = TestInpaintDataset(opt)
    num_img = TestInpaintDataset(opt).__len__()
    # Load to dataloader
    dataloader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.num_workers, pin_memory=True)
    
    print("Doing validation!")
    for batch_idx, (img, mask) in enumerate(dataloader):
        # For metrics
        ssim_total = 0
        psnr_total = 0
        mae_total = 0
        
        # Load model to device
        img = img.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_generated = img * (1 - mask) + first_out * mask        # in range [0, 1]
        second_generated = img * (1 - mask) + second_out * mask      # in range [0, 1]    

        # Calculate metrics

        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        img_list = [second_generated]
        name_list = ["second_out"]

        # MAE
        mae = compare_mae(img, second_generated)
        mae_total += mae
        # Psnr
        psnr_v = psnr(second_generated, img)
        psnr_total += psnr_v
        # SSIM
        ssim_v = ssim(second_generated, img)
        ssim_total += ssim_v

        # Save image
        save_sample_png(sample_folder = opt.results_path, sample_name = '%d' % (batch_idx + 1), 
                img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        print("Inpainting finished !!!")

    print("MAE value: {}".format(mae_total/num_img))
    print("PSNR value: {}".format(psnr_total/num_img))
    print("SSIM value: {}".format(ssim_total/num_img))


if __name__=='__main__':
    opt = TestOptions().parse()
    Inpain_Test(opt)