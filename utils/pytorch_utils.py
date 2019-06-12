
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
from utils import logger

use_cuda = torch.cuda.is_available()


# utility
def to_var(x, dtype=None):
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    elif type(x) is list:
        x = torch.from_numpy(np.array(x, dtype=dtype))

    if use_cuda:
        x = x.cuda()

    return Variable(x)


# optimization
# reference: http://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
def adjusting_learning_rate(optimizer, factor=.5, min_lr=0.00001):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr*factor, min_lr)
        param_group['lr'] = new_lr
        logger.info('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))


def lr_annealing_function(step, start=0, end=1, r=0.9999, type="exp"):
    if type == "exp":
        lr = start - (start - end) * (1 - math.pow(r, step))
    else:
        print("not available %s annealing" % type)

    return lr


def update_lr(optimizer, new_lr):
    old_lr = optimizer.param_groups[0]['lr']
    # logger.info("adjusting learning rate from %.6f to %.6f" % (old_lr, new_lr))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lr


def transformer_learning_rate(optimizer, model_dim, step_num, warmup_steps=4000):
    for i, param_group in enumerate(optimizer.param_groups):
        new_lr = model_dim**(-0.5) * min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
        old_lr = float(param_group['lr'])
        # new_lr = max(old_lr*factor, min_lr)
        param_group['lr'] = new_lr
        logger.info('adjusting learning rate from  %.6f to %.6f' % (old_lr, new_lr))


# model save and loading
def load_model(asset_path, model, optimizer, restore_epoch=0):
    if os.path.isfile(os.path.join(asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch)):
        checkpoint = torch.load(os.path.join(asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_step = checkpoint['current_step']
        logger.info("restore model with %d epoch" % restore_epoch)
    else:
        logger.info("no checkpoint with %d epoch" % restore_epoch)
        current_step = 0

    return model, optimizer, current_step


# class weighted_BCELoss(Module):
#     def __init__(self, mode):
#         self.mode = mode
#
#     def forward(self, input, target, weight=10):
#         if not (input.size() == target.size()):
#             raise ValueError("Target and input must have the same size. target size ({}) "
#                              "!= input size ({})".format(target.size(), input.size()))
#         loss_matrix = - (torch.mul(target, input.log()) + torch.mul(1 - target, (1 - input).log()))
#         one_matrix = Variable(torch.ones(input.size()))
#         if use_cuda:
#             one_matrix = one_matrix.cuda()
#         if self.mode == 'one':
#             weight_matrix = (weight - 1) * target + one_matrix
#         elif self.mode == 'pitch':
#
#         weighted_loss_matrix = torch.mul(loss_matrix, weight_matrix)
#         return torch.mean(weighted_loss_matrix)

# loss
def weighted_binary_cross_entropy(output, target, weights=None, eps=1e-12):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output + eps)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + eps))
    else:
        loss = target * torch.log(output + eps) + (1 - target) * torch.log(1 - output + eps)

    return torch.neg(torch.mean(loss))


def kl_divergence(mu, sig, num_latent_group=0, freebits_ratio=2., p_mu=None, p_sigma=None, eps=1e-8):
    # calculate kl divergence between two normal distribution
    # mu, sig, p_mu, p_sigma: batch_size * latent_size
    batch_size = mu.size(0)
    latent_size = mu.size(1)

    mu_square = mu * mu
    sig_square = sig * sig

    if p_mu is None:
        kl = 0.5 * (mu_square + sig_square - torch.log(sig_square + eps) - 1)
    else:
        p_sig_square = p_sigma * p_sigma
        p_mu_diff_square = (mu - p_mu) * (mu - p_mu)
        kl = (sig_square + p_mu_diff_square)/(2*p_sig_square)
        kl += torch.log(p_sigma/sig + eps)
        kl -= 0.5

    if num_latent_group == 0:
        kl = torch.sum(kl) / batch_size
    else:
        group_size = latent_size // num_latent_group
        kl = kl.mean(0)  # mean along batch dimension
        kl = kl.view(-1, group_size).sum(1)  # summation along group dimension
        kl = torch.clamp(kl, min=freebits_ratio)  # clipping kl value
        kl = kl.sum()

    return kl


def vae_loss(target, prediction, mu, sig,
             num_latent_group=0, freebits_ratio=2., kl_ratio=1., p_mu=None, p_sigma=None):

    rec_loss = F.binary_cross_entropy(prediction, target)
    kl_loss = kl_divergence(mu, sig, num_latent_group, freebits_ratio, p_mu, p_sigma)

    total_loss = rec_loss + kl_ratio * kl_loss

    return total_loss, rec_loss, kl_loss
