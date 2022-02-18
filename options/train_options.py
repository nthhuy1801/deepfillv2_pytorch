import argparse
import os


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Network parameters
        self.parser.add_argument('--in_channels', type=int, default=4, help='The input of RGB image + 1 channel of mask')
        self.parser.add_argument('--out_channels', type=int, default=3, help='Output RGB image')
        self.parser.add_argument('--latent_channels', type=int, default=64, help='Latent channels')
        self.parser.add_argument('--pad_type', type=str, default='zero', help='Padding types: zero, reflect, replicate')
        self.parser.add_argument('--activation', type=str, default='LeakyReLU', help='Activation types: ReLU, LeakyReLU, ELU, SELU, PReLU, Tanh, Sigmoid, none')
        self.parser.add_argument('--norm', type=str, default = 'in', help = 'normalization type')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='The initialized gain')
        # Dataset parameter
        self.parser.add_argument('--baseroot', type=str, default='/home/huynth/deepfillv2_thesis/data/place2', help='the training image folder')
        self.parser.add_argument('--mask_type', type=str, default='free_form', help='Mask type: free form, bounding box')
        self.parser.add_argument('--image_size', type=int, default=256, help='Resize image in training set to this size')
        self.parser.add_argument('--dataset', type=str, default='celeba-hq', help="Dataset name for training")
        self.parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
        self.parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
        self.parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
        self.parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
        self.parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
        # Training parameters
        self.parser.add_argument('--epochs', type=int, default=100, help='Numbers of epochs training')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size uses during training')
        self.parser.add_argument('--lr_g', type=float, default=1e-4, help='Hệ số learning rate của hàm tối ưu Adam của generator')
        self.parser.add_argument('--lr_d', type=float, default=4e-4, help='Hệ số learning rate của hàm tối ưu Adam của discriminator')
        self.parser.add_argument('--b1', type=float, default=0.5, help='Adam: beta 1')
        self.parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay of Adam optimizer')
        self.parser.add_argument('--lr_decrease_epoch', type=int, default=10, help='lr decrease at certain epoch and its multiple')
        self.parser.add_argument('--lr_decrease_factor', type=float, default=0.5, help='lr decrease factor, for classification default 0.1')
        self.parser.add_argument('--lambda_l1', type=float, default=100, help='the parameter of L1Loss')
        self.parser.add_argument('--lambda_perceptual', type=float, default=10, help='the parameter of FML1Loss (perceptual loss)')
        self.parser.add_argument('--lambda_gan', type=float, default=1, help='the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')
        # Training
        self.parser.add_argument('--save_path', type=str, default='./save_models', help='thư mục lưu model')
        self.parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
        self.parser.add_argument('--gpu_ids', type=str, default="0", help='GPU ids to be used: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--load_name', type = str, default = '', help = 'load model name')
        self.parser.add_argument('--gan_type', type=str, default='WGAN', help='the type of GAN for training')
        self.parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.print_options(self.opt)
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(str(id))

        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        if not os.path.isdir(self.opt.baseroot):
            os.mkdir(self.opt.baseroot)

        args = vars(self.opt)

        print("-" * 20 + " Options " + "-" * 20)
        for k, v in sorted(args.items()):
            print(str(k), ":", str(v))
        print("-" * 20 + " End " + "-" * 20)

        return self.opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

if __name__=='__main__':
    options = TrainOptions()
    args = options.parse()