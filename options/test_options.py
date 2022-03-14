import os
import argparse

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--baseroot', type=str, default='/home/huynth/deepfillv2_thesis/data/place2', help='the training image folder')
        self.parser.add_argument('--maskroot', type=str, default='home/huynth/deepfillv2_thesis/mask')
        self.parser.add_argument('--mask_type', type=str, default='free_form', help='Mask type: free form, bounding box')
        self.parser.add_argument('--image_size', type=int, default=256, help='Resize image in training set to this size')
        self.parser.add_argument('--dataset', type=str, default='celeba-hq', help="Dataset name for training")
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')

        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size uses during training')
        self.parser.add_argument('--load_name', type = str, default = './save_models/deepfillv2_G_epoch7_batchsize16.pth', help = 'test model name')
        # Network parameters
        self.parser.add_argument('--epochs', type=int, default=40, help='Numbers of epochs training')
        self.parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
        self.parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
        self.parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
        self.parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
        self.parser.add_argument('--activation', type = str, default = 'ELU', help = 'the activation type')
        self.parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
        self.parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
        self.parser.add_argument('--results_path', type = str, default = './results', help = 'testing samples path that is a folder')
        self.initialized = True

    def parse(self):
        if not self.initialized :
            self.initialize()

        self.opt = self.parser.parse_args()

        if not os.path.isdir(self.opt.baseroot) :
            os.mkdir(self.opt.baseroot)

        args = vars (self.opt)

        print ("-"*20 + " Options " + "-"*20)
        for k, v in sorted (args.items()) :
            print (str(k), ":", str(v))
        print ("-"*20 + " End " + "-"*20)

        return self.opt


if __name__=='__main__':
    test_option = TestOptions()
    args = test_option.parse()