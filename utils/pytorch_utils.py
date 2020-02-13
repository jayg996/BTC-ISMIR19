
import torch
import numpy as np
import os
import math
from utils import logger

use_cuda = torch.cuda.is_available()


# optimization
# reference: http://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
def adjusting_learning_rate(optimizer, factor=.5, min_lr=0.00001):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr * factor, min_lr)
        param_group['lr'] = new_lr
        logger.info('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))


# model save and loading
def load_model(asset_path, model, optimizer, restore_epoch=0):
    if os.path.isfile(os.path.join(asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch), map_location=lambda storage, loc: storage):
        checkpoint = torch.load(os.path.join(asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_step = checkpoint['current_step']
        logger.info("restore model with %d epoch" % restore_epoch)
    else:
        logger.info("no checkpoint with %d epoch" % restore_epoch)
        current_step = 0

    return model, optimizer, current_step
