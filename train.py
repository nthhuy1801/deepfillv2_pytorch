import sys
from trainer.trainer import *
from options.train_options import TrainOptions

opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

model = WGANTrainer(opt)