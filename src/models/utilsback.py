import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT plan.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
       

        return pi

    def M(self, C, u, v):
        # "Modified cost for logarithmic updates"
        # "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2).cuda()
        y_lin = y.unsqueeze(-3).cuda()
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
    

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

# def mixup_process(out, target_reweighted, lam):
#     indices = np.random.permutation(out.size(0))
#     out = out*lam + out[indices]*(1-lam)
#     target_shuffled_onehot = target_reweighted[indices]
#     target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
#     return out, target_reweighted

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(1. - lam)
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# def mixup_process(out, target_reweighted, lam, cutmix=False):
#     indices = np.random.permutation(out.size(0))
#     if cutmix:
#         target_a = target_reweighted
#         target_b = target_reweighted[indices]
#         bbx1, bby1, bbx2, bby2 = rand_bbox(out.size(), lam)
#         out[:, :, bbx1:bbx2, bby1:bby2] = out[indices, :, bbx1:bbx2, bby1:bby2]
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (out.size()[-1] * out.size()[-2]))
#         target_reweighted = target_a * lam + target_b * (1 - lam)
#     else:
#         out = out * lam + out[indices] * (1-lam)
#         target_shuffled_onehot = target_reweighted[indices]
#         target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
#     return out, target_reweighted

def mixup_process(out, target_reweighted, lam, cutmix=False, align_mixup=False):
    indices = np.random.permutation(out.size(0))
    if cutmix:
        target_a = target_reweighted
        target_b = target_reweighted[indices]
        bbx1, bby1, bbx2, bby2 = rand_bbox(out.size(), lam)
        out[:, :, bbx1:bbx2, bby1:bby2] = out[indices, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (out.size()[-1] * out.size()[-2]))
        target_reweighted = target_a * lam + target_b * (1 - lam)

    if align_mixup:
    # out shape = batch_size x 512 x 4 x 4 (cifar10/100)

        indices = np.random.permutation(out.size(0))
        feat1 = out.view(out.shape[0], out.shape[1], -1) # batch_size x 512 x 16
        feat2 = out[indices].view(out.shape[0], out.shape[1], -1) # batch_size x 512 x 16
        
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        P = sinkhorn(feat1.permute(0,2,1), feat2.permute(0,2,1)).detach()  # optimal plan batch x 16 x 16
        
        P = P*(out.size(2)*out.size(3)) # assignment matrix 

        align_mix = random.randint(0,1) # uniformly choose at random, which alignmix to perform
    
        if (align_mix == 0):
            # \tilde{A} = A'R^{T}
            f1 = torch.matmul(feat2, P.permute(0,2,1).cuda()).view(out.shape) 
            final = feat1.view(out.shape)*lam + f1*(1-lam)

        elif (align_mix == 1):
            # \tilde{A}' = AR
            f2 = torch.matmul(feat1, P.cuda()).view(out.shape).cuda()
            final = f2*lam + feat2.view(out.shape)*(1-lam)

        y_a, y_b = target_reweighted, target_reweighted[indices]
        y_a = y_a.cuda()
        y_b = y_b.cuda()
        return final, y_a * lam  + y_b * (1-lam)
    else:
        target_reweighted = target_reweighted.cuda()
        out = out * lam + out[indices] * (1-lam)
        target_shuffled_onehot = target_reweighted[indices]
        target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def apply(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()
        img = img * mask

        return img



def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    #return Variable(y_onehot.cuda(),requires_grad=False)
    return y_onehot

