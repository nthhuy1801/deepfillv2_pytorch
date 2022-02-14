import os
import argparse

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--baseroot', type=str, default='/home/huynth/deepfillv2_thesis/data/place2', help='the training image folder')
        self.parser.add_argument('--mask_type', type=str, default='free_form', help='Mask type: free form, bounding box')
        self.parser.add_argument('--image_size', type=int, default=256, help='Resize image in training set to this size')
        self.parser.add_argument('--dataset', type=str, default='celeba-hq', help="Dataset name for training")

        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size uses during training')
        self.parser.add_argument('--load_name', type = str, default = 'model.pth', help = 'test model name')
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