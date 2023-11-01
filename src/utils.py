import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from rich import print as print_color
from collections import namedtuple
from typing import Dict, List, Optional, Union, Set


record_fields = ['epoch', 'loss', 'top1', 'top5', 'error1', 'error5', 'recall1', 'recall5']

record_type = namedtuple('Record',
                    field_names=record_fields,
                    defaults=(None, ) * len(record_fields))

record_type_lt = namedtuple('Record',
                    field_names=['epoch', 'loss', 'top1', 'top5', 'error1', 'error5', 'recall1', 'recall5','many_acc','medium_acc','few_acc'],
                    defaults=(None, ) * 2)

key_prompt = dict(
    epoch="[bold white]Epoch",
    loss="[bold bright_red]Loss",
    top1="[bold bright_blue]Prec@1",
    top5="[bold bright_cyan]Prec@5",
    error1="[bold blue]Error@1",
    error5="[bold cyan]Error@5",
    recall1="[bold green]Recall@1",
    recall5="[bold green4]Recall@5",
    many_acc="[bold green4]Many_acc@1", 
    medium_acc="[bold green4]Medium_acc@1",
    few_acc="[bold green4]Few_acc@1",
)

# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = torch.sqrt(1. - lam)
#     cut_w = (W * cut_rat).int()
#     cut_h = (H * cut_rat).int()

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
#     bby1 = torch.clamp(cy - cut_h // 2, 0, H)
#     bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
#     bby2 = torch.clamp(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2

# def class_aware_indices(out, target):
#     sorted_target = target.sort()
#     di = dict()
#     for label,idx in zip(sorted_target[0].tolist(),sorted_target[1].tolist()):
#         if label in di.keys():
#             di[label].append(idx)
#         else:
#             di[label] = [idx]

#     unique_labels = torch.unique(sorted_target[0])
#     additional_random_size = out.size(0) - len(unique_labels)
#     indices = np.random.randint(0,len(unique_labels), additional_random_size) # batch_size - unique한 label들의 개수
#     indices = unique_labels[indices]
#     indices = torch.cat((unique_labels, indices))
#     indices = indices[torch.randperm(out.size(0))]
    
#     for idx, val in enumerate(indices):
#         indices[idx] = random.choice(di[val.item()]) #random choice가 나은지 pop이 나은지 모르곘음

#     return indices



# def mixup_process(out, target, target_reweighted, lam, cutmix=False, class_aware=False):
#     if class_aware:
#         indices = class_aware_indices(out,target)
#     else:
#         indices = np.random.permutation(out.size(0))
#     if cutmix:
#         target_a = target_reweighted
#         target_b = target_reweighted[indices]
#         bbx1, bby1, bbx2, bby2 = rand_bbox(out.size(), lam)
#         out[:, :, bbx1:bbx2, bby1:bby2] = out[indices, :, bbx1:bbx2, bby1:bby2]
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (out.size()[-1] * out.size()[-2]))
#         target_reweighted = target_a * lam + target_b * (1 - lam)
#     else:
#         out = out * lam + out[indices] * (1-lam) #shuffle 부분
#         target_shuffled_onehot = target_reweighted[indices]
#         target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
#     return out, target_reweighted

class Record(record_type):
    "A record class for keep scores and compare"
    compare_key: Optional[str] = None

    def __repr__(self):
        keys = record_type._fields
        message = [
            f"{key_prompt[key]}: {val:7.4f}" for key in keys if (val := getattr(self, key)) is not None
        ]
        return ' '.join(message)

    def __gt__(self, other):
        if self.compare_key is None:
            return self.epoch > other.epoch
        else:
            if self.compare_key in ["loss", "error1", "error5"]:
                comp = getattr(self, self.compare_key, 0) < getattr(other, self.compare_key, 0)
            else:
                comp = getattr(self, self.compare_key, 0) > getattr(other, self.compare_key, 0)
            return comp

    def __eq__(self, other):
        return self.epoch == other.epoch

    def __hash__(self):
        return hash(self.epoch)

class RecordLT(record_type_lt):
    "A record class for keep scores and compare"
    compare_key: Optional[str] = None

    def __repr__(self):
        keys = record_type_lt._fields
        message = [
            f"{key_prompt[key]}: {val:7.4f}" for key in keys if (val := getattr(self, key)) is not None
        ]
        return ' '.join(message)

    def __gt__(self, other):
        if self.compare_key is None:
            return self.epoch > other.epoch
        else:
            if self.compare_key in ["loss", "error1", "error5"]:
                comp = getattr(self, self.compare_key, 0) < getattr(other, self.compare_key, 0)
            else:
                comp = getattr(self, self.compare_key, 0) > getattr(other, self.compare_key, 0)
            return comp

    def __eq__(self, other):
        return self.epoch == other.epoch

    def __hash__(self):
        return hash(self.epoch)


class Recorder:
    def __init__(self, lt=False):
        self.lt = lt
        self.records: List[Record] = list()
        self.cur_epoch: int = -1
        self.val_keys = [field for field in Record._fields if field != 'epoch']
        if lt:
            self.val_keys+=['many_acc', 'medium_acc', 'few_acc']

    def __call__(self,
                 epoch: int,
                 valid_res: Dict[str, Union[float, int]]) -> None:
        self.cur_epoch = epoch
        
        if self.lt:
            self.records.append(
                RecordLT(epoch=epoch, **valid_res)
            )

            keep_record: Set[RecordLT] = set()
            print_color(f"[bold]Current Epoch: [white]{self.cur_epoch}")
            for key in self.val_keys:
                RecordLT.compare_key = key
                key_record = max(self.records)
                print_color(f"[bold green]Record by {key:7}", end='')
                print_color(f"[bold white]==> {key_record}")
                keep_record.add(key_record)
            self.records = list(keep_record)
        else:
            self.records.append(
                Record(epoch=epoch, **valid_res)
            )

            keep_record: Set[Record] = set()
            print_color(f"[bold]Current Epoch: [white]{self.cur_epoch}")
            for key in self.val_keys:
                Record.compare_key = key
                key_record = max(self.records)
                print_color(f"[bold green]Record by {key:7}", end='')
                print_color(f"[bold white]==> {key_record}")
                keep_record.add(key_record)
            self.records = list(keep_record)


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
    if self.count != 0:
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

def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    if inp.dtype != "torch.int64":
        inp = inp.type(torch.int64)
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    
    #t1 = target.data.cpu().numpy()
    #t2 = target[indices].data.cpu().numpy()
    #print (np.sum(t1==t2))
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


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

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


def create_val_folder(data_set_path):
    """
    Used for Tiny-imagenet dataset
    Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
    This method is responsible for separating validation images into separate sub folders,
    so that test and val data can be read by the pytorch dataloaders
    """
    path = os.path.join(data_set_path, 'val/images')  # path where validation data is present now
    filename = os.path.join(data_set_path, 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

if __name__ == "__main__":
    create_val_folder('data/tiny-imagenet-200')  # Call method to create validation image folders



