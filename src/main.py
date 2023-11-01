#!/usr/bin/env python
from __future__ import division

import os, sys, shutil, time, random
import argparse
from distutils.dir_util import copy_tree
from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import precision_recall
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import *
from pathlib import Path
import models
# from data import dataloader

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import numpy as np
from collections import OrderedDict, Counter
from load_data  import *
from helpers import *
from analytical_helper_script import run_test_with_mixup

import wandb
from text_manifold_aug import TextManiA


model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar100-lt', choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet-200', 'cifar10-lt', 'cifar100-lt', 'imagenet-lt'])
parser.add_argument('--data_dir', type = str, default = 'cifar100', help='file where results are to be written')
parser.add_argument('--root_dir', type = str, default = 'experiments', help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=500, metavar='NL', help='labels_per_class')
parser.add_argument('--valid_labels_per_class', type=int, default=0, metavar='NL', help='validation labels_per_class')

parser.add_argument('--ibf', type=int, default=100, choices=(100,50,10))
parser.add_argument('--arch', metavar='ARCH', default='resnet34', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet34)')
parser.add_argument('--initial_channels', type=int, default=64, choices=(16,64))
parser.add_argument('--aug_group', type=str, default="all", choices=['all', 'midfew','few'])

# Optimization options
parser.add_argument('--optimizer', type=str, default="SGD", help='type of optimizer')
parser.add_argument('--sampler', type=str, default=False, help='classAware')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--train', type=str, default = 'vanilla', choices =['vanilla','mixup',
                                                                        'mixup_hidden','cutout',
                                                                        'cutmix', 'textmania', 'noise_vec',
                                                                        'mixup_hidden_textmania', 'mixup_textmania',
                                                                        'cutmix_textmania', 'cutout_textmania'])
parser.add_argument('--mixup_alpha', type=float, default=0.0, help='alpha parameter for mixup')
parser.add_argument('--cutout', type=int, default=16, help='size of cut out')
parser.add_argument('--dropout', action='store_true', default=False, help='whether to use dropout or not in final layer')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1.')
parser.add_argument('--beta2', type=float, default=0.999, help='Beta2.')

parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--adv_unpre', action='store_true', default=False, help= 'the adversarial examples will be calculated on real input space (not preprocessed)')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[50,100,150], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', default=43, type=int, help='manual seed')
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--job_id', type=str, default='')
# wandb
parser.add_argument('--project', default='long-tail-aug', type=str, help='name for wandb project')
parser.add_argument('--wandb', default='test', type=str, help='name for wandb logging')
parser.add_argument('--max_sample', type=int, default=30, help='the number of sampled text augmentation')
parser.add_argument('--scale', type=float, default=0.1, help='alpha scale minimum value')
parser.add_argument('--nsample', default=True, action='store_true', help='true -> use random n sample from diff vector')
parser.add_argument('--multi_prompts', type=int, default=1, help='type of multiple prompts')
parser.add_argument('--bce_loss', default=False, action='store_true', help='use bce loss instead ce loss')
parser.add_argument('--l1_const', default=False, action='store_true', help='use l1 constraint for text manifold')
parser.add_argument('--attr', type=str, nargs='+', default=['color'])
parser.add_argument('--group', type=str, default='debug')
parser.add_argument('--cutmix_prob', type=float, default=0.5)
parser.add_argument('--multi_attr', default=False, action='store_true', help='use of color & size attribute')
parser.add_argument('--scale_max', type=float, default=1.0, help='alpha scale maximum value')

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

out_str = str(args)
print(out_str)

ROOT = Path(__file__).resolve().parents[1]
cudnn.benchmark = True


def experiment_name_non_mnist(dataset='cifar10',
                    epochs=400,
                    train = 'vanilla',
                    job_id=None,
                    ibf=None,
                    add_name=''):
    exp_name = dataset
    exp_name += '_train_'+str(train)
    exp_name += '_eph_'+str(epochs)
    exp_name += '_ibf_'+str(ibf)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    print('experiement name: ' + exp_name)
    return exp_name


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, num_classes, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(-1).expand_as(pred))

    acc = []
    recalls = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc.append(correct_k.mul_(100.0 / batch_size))
        _, recall = precision_recall(output, target, "macro",
                                    num_classes=num_classes, top_k=k, multiclass=True)
        recalls.append(recall.mul_(100.0 / batch_size))
    return acc, recalls


def shot_acc(preds, labels, train_class_count, many_shot_thr=100, low_shot_thr=20):
    _, preds = preds.topk(1, 1, True, True)
    preds = preds.squeeze(-1)
    
    shot_cnt_stats = {
        "many": [many_shot_thr - 1, max(train_class_count), 0, 0, 0.],
        "median": [low_shot_thr, many_shot_thr - 1, 0, 0, 0.],
        "low": [0, low_shot_thr, 0, 0, 0.],
        "10-shot": [0, 10, 0, 0, 0.],
        "5-shot": [0, 5, 0, 0, 0.],
    }
    for l in torch.unique(labels):
        class_correct = torch.sum((preds[labels == l] == labels[labels == l])).item()
        test_class_count = len(labels[labels == l])
        for stat_name in shot_cnt_stats:
            stat_info = shot_cnt_stats[stat_name]
            if train_class_count[l] > stat_info[0] and train_class_count[l] <= stat_info[1]:
                stat_info[2] += class_correct
                stat_info[3] += test_class_count
    for stat_name in shot_cnt_stats:
        shot_cnt_stats[stat_name][-1] = shot_cnt_stats[stat_name][2] / shot_cnt_stats[stat_name][3] * \
                                        100.0 if shot_cnt_stats[stat_name][3] != 0 else 0.
    return shot_cnt_stats

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
criterion = nn.CrossEntropyLoss().cuda()
mse_loss = nn.MSELoss().cuda()


def train(train_loader, model, optimizer, epoch, args, log, textmania=None, label_groups=None, aug_group=args.aug_group):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    recall1 = AverageMeter()
    recall5 = AverageMeter()

    # switch to train mode
    model.train()
    projection = None

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.long()
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time() - end)

        if args.train == 'mixup':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, mixup_hidden=False, mixup=True, mixup_alpha=args.mixup_alpha)
            loss = bce_loss(softmax(output), reweighted_target) 
        
        elif args.train == "cutmix":
            r = np.random.rand(1)
            cutmix = False
            if args.mixup_alpha > 0 and r < args.cutmix_prob:
                cutmix = True

            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, cutmix=cutmix, mixup_alpha=args.mixup_alpha)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == "cutmix_textmania":
            r = np.random.rand(1)
            cutmix = False
            if args.mixup_alpha > 0 and r < args.cutmix_prob:
                cutmix = True

            input_var, target_var = Variable(input), Variable(target)
            base_feats, diff_feats, aug_target = textmania(target_var, with_mixup=True)
            ori_out, aug_out, ori_targets, aug_targets = model(input_var, target=target_var, diff_vec=diff_feats,
                                              cutmix=cutmix, mixup_alpha=args.mixup_alpha, aug_target=aug_target,)
            loss = bce_loss(softmax(ori_out), ori_targets)
            loss += bce_loss(softmax(aug_out), aug_targets)
            output = ori_out
            target = target_var

        elif args.train == 'mixup_textmania':
            input_var, target_var = Variable(input), Variable(target)
            base_feats, diff_feats, aug_target = textmania(target_var, with_mixup=True)

            ori_out, aug_out, ori_targets, aug_targets = model(input_var,
                                                                target=target_var, diff_vec=diff_feats, aug_target=aug_target,
                                                                mixup_hidden=False, mixup=True, mixup_alpha=args.mixup_alpha)

            loss = bce_loss(softmax(ori_out), ori_targets)
            loss += bce_loss(softmax(aug_out), aug_targets)
            output = ori_out
            target = target_var

        elif args.train == 'cutout':
            cutout = Cutout(1, args.cutout)
            cut_input = cutout.apply(input)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            cut_input_var = torch.autograd.Variable(cut_input)
            output, reweighted_target = model(cut_input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'cutout_textmania':
            cutout = Cutout(1, args.cutout)
            cut_input = cutout.apply(input)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            cut_input_var = torch.autograd.Variable(cut_input)
            base_feats, diff_feats, aug_labels = textmania(target_var)
            ori_out, aug_out = model(cut_input_var, diff_vec=diff_feats)

            ori_labels = to_one_hot(target, args.num_classes).to(input_var.device)
            aug_labels = to_one_hot(aug_labels, args.num_classes).to(input_var.device)
            loss = bce_loss(softmax(ori_out), ori_labels)
            loss += bce_loss(softmax(aug_out), aug_labels)
            output = ori_out
            target = target_var

        elif args.train == 'mixup_hidden':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha)
            loss = bce_loss(softmax(output), reweighted_target)  # mixup_criterion(target_a, target_b, lam)

        elif args.train == 'mixup_hidden_textmania':
            input_var, target_var = Variable(input), Variable(target)
            base_feats, diff_feats, aug_target = textmania(target_var, with_mixup=True)

            ori_out, aug_out, ori_targets, aug_targets = model(input_var,
                                                                target=target_var, diff_vec=diff_feats, aug_target=aug_target,
                                                                mixup_hidden=True, mixup_alpha=args.mixup_alpha)

            loss = bce_loss(softmax(ori_out), ori_targets)
            loss += bce_loss(softmax(aug_out), aug_targets)
            output = ori_out
            target = target_var

        elif args.train == 'vanilla':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'textmania':
            input_var, target_var = Variable(input), Variable(target)
            base_feats, diff_feats, aug_labels = textmania(target_var)

            ori_out, aug_out = model(input_var,
                                    diff_vec=diff_feats,
                                    scale=args.scale, max_scale=args.scale_max)

            ori_labels = to_one_hot(target, args.num_classes).to(input_var.device)
            aug_labels = to_one_hot(aug_labels, args.num_classes).to(input_var.device)
            loss = bce_loss(softmax(ori_out), ori_labels)
            loss += bce_loss(softmax(aug_out), aug_labels)
            output = ori_out
            target = target_var

        

        # measure accuracy and record loss
        acc, recall = accuracy(output, target, args.num_classes, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(acc[0].item(), output.size(0))
        top5.update(acc[1].item(), output.size(0))
        recall1.update(recall[0].item(), output.size(0))
        recall5.update(recall[1].item(), output.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                'Recall@1 {recall1.val:.3f} ({recall1.avg:.3f})   '
                'Recall@5 {recall5.val:.3f} ({recall5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, recall1=recall1, recall5=recall5)
                + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, top5.avg, losses.avg, recall1.avg, recall5.avg


def validate(val_loader, model, log, args, cls_num_list=None, label_groups=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    recall1 = AverageMeter()
    recall5 = AverageMeter()
    many_acc = AverageMeter()
    medium_acc = AverageMeter()
    few_acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()

        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var).mean()

    # measure accuracy and record loss
        acc, recall = accuracy(output, target, args.num_classes, topk=(1, 5))
        # if training_labels is not None:
            # many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(output, target, training_labels, acc_per_cls=False)
        if label_groups is not None:
            status = shot_acc(output, target, cls_num_list, many_shot_thr=100, low_shot_thr=20)
            many_acc_top1 = status['many'][-1]
            medium_acc_top1= status['median'][-1]
            few_acc_top1= status['low'][-1]
            many_count = status['many'][3]
            medium_count= status['median'][3]
            few_count = status['low'][3]
        
        else:
            many_acc_top1 = None
            medium_acc_top1 = None
            few_acc_top1 = None

        losses.update(loss.item(), input.size(0))
        top1.update(acc[0].item(), input.size(0))
        top5.update(acc[1].item(), input.size(0))
        recall1.update(recall[0], input.size(0))
        recall5.update(recall[1], input.size(0))

        if many_acc_top1 is not None:
            many_acc.update(many_acc_top1, many_count)
        if medium_acc_top1 is not None:
            medium_acc.update(medium_acc_top1, medium_count)
        if few_acc_top1 is not None:
            few_acc.update(few_acc_top1, few_count)


    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)


    if label_groups:
        valid_res = dict(
            top1=top1.avg,
            top5=top5.avg,
            error1=100-top1.avg,
            error5=100-top5.avg,
            loss=losses.avg,
            recall1=recall1.avg,
            recall5=recall5.avg,
            many_acc=many_acc.avg,
            medium_acc=medium_acc.avg,
            few_acc=few_acc.avg,
        )
    else:
        valid_res = dict(
            top1=top1.avg,
            top5=top5.avg,
            error1=100-top1.avg,
            error5=100-top5.avg,
            loss=losses.avg,
            recall1=recall1.avg,
            recall5=recall5.avg,
        )
    return valid_res


best_acc = 0

def divide_group(num_per_cls_dict, dataset):
    label_groups = {"many":[], "medium":[], "few":[]}

    if dataset == "cifar100-lt":
        for idx, num_img in num_per_cls_dict.items():
            if num_img >100:
                label_groups["many"].append(idx)
            elif num_img<=100 and num_img>=20:
                label_groups["medium"].append(idx)
            else:
                label_groups["few"].append(idx)

    elif dataset == "cifar10-lt":
        for idx, num_img in num_per_cls_dict.items():
            if num_img >1000:
                label_groups["many"].append(idx)
            elif num_img<=1000 and num_img>=200:
                label_groups["medium"].append(idx)
            else:
                label_groups["few"].append(idx)

    elif dataset == "imagenet-lt":
        for idx, num_img in enumerate(num_per_cls_dict):
            if num_img >1000:
                label_groups["many"].append(idx)
            elif num_img<=1000 and num_img>=200:
                label_groups["medium"].append(idx)
            else:
                label_groups["few"].append(idx)


    return label_groups

def seed_everything(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


adjective = dict(
    color='colored',
    size='sized'
)

def main():
    # wandb setting
    wandb.init(project=args.project)
    wandb.run.name = args.wandb
    wandb.config.update(args)

    # validation recoder
    if args.dataset in ["cifar100-lt","cifar10-lt", "imagenet-lt"]:
        lt=True
    else:
        lt=False

    val_recorder = Recorder(lt = lt)

    ### set up the experiment directories########
    exp_name = experiment_name_non_mnist(dataset=args.dataset,
                    epochs=args.epochs,
                    train = args.train,
                    job_id=args.job_id,
                    ibf=args.ibf,
                    add_name=args.add_name)

    exp_dir = args.root_dir+exp_name

    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    copy_script_to_folder(os.path.abspath(__file__), exp_dir)

    result_png_path = os.path.join(exp_dir, 'results.png')


    global best_acc

    # seed fix
    seed_everything(args.manualSeed)

    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(exp_dir), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    if args.adv_unpre:
        per_img_std = True
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset_unpre(args.data_aug, args.batch_size, 2 ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class)
    else:
        per_img_std = False
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, 2 ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class, imbalance_factor = args.ibf)

    # for long-tail divide target into groups
    if args.dataset in ["cifar100-lt", "cifar10-lt"]:
        label_groups = divide_group(train_loader.dataset.num_per_cls_dict, args.dataset)
    else:
        train_labels = None
        label_groups = None

    if args.dataset == 'tiny-imagenet-200':
        stride = 2
    else:
        stride = 1
    #train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, 2, args.dataset, args.data_dir, 0.0, labels_per_class=5000)
    print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](num_classes,args.dropout,per_img_std, stride).cuda()
    print_log("=> network :\n {}".format(net), log)
    args.num_classes = num_classes

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)


    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, log, args, cls_num_list = train_loader.cls_num_list, label_groups = label_groups)
        print("Validation finished.")
        return

    if "textmania" in args.train:
        if args.multi_prompts == 1:
            prompts = ROOT / 'preprocessing/prompt.txt'
            prompts = prompts.read_text().split('\n')
        elif args.multi_prompts == 2:
            prompts = [
                'a photo of a',
                'a photo of the',
                'a picture of a',
                'a picture of the',
                'an image of a',
                'an image of the',
            ]
        else:
            prompts = 'a photo of the'

        attr = args.attr
        adj = [adjective[a] for a in attr]

        if label_groups is None:
            aug_group=None
        elif args.aug_group=="all":
            aug_group = label_groups["many"]+label_groups["medium"]+label_groups["few"]
        elif args.aug_group=="midfew":
            aug_group = label_groups["medium"]+label_groups["few"]
        elif args.aug_group=="few":
            aug_group = label_groups["few"]
        
        textmania = TextManiA(args, prompts, attr, adj,
                                    args.max_sample, args.nsample, args.scale)
    else:
        textmania = None

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    # Main loop
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []


    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        tr_acc, tr_acc5, tr_los, tr_recall1, tr_recall5 = train(train_loader, net, optimizer, epoch, args, log, textmania, label_groups= label_groups, aug_group=args.aug_group)

        # evaluate on validation set
        # valid_res = validate(test_loader, net, log, args, train_labels)
        if lt:
            valid_res = validate(test_loader, net, log, args, cls_num_list = train_loader.cls_num_list, label_groups=label_groups)
        else:
            valid_res = validate(test_loader, net, log, args, cls_num_list = None, label_groups=None)
        val_recorder(epoch, valid_res)

        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(valid_res['loss'])
        test_acc.append(valid_res['top1'])

        wandb.log({
            "train loss": tr_los,
            "train acc@1": tr_acc,
            "train acc@5": tr_acc5,
            "val loss": valid_res['loss'],
            "val acc": valid_res['top1'],
            "train_recall@1": tr_recall1,
            "train_recall@5": tr_recall5,
        })
        dummy = recorder.update(epoch, tr_los, tr_acc, valid_res['loss'], valid_res['top1'])

        is_best = False
        if valid_res['top1'] > best_acc:
            is_best = True
            best_acc = valid_res['top1']

        save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
        }, is_best, exp_dir, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)

        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc'] = train_acc
        train_log['test_loss'] = test_loss
        train_log['test_acc'] = test_acc

        pickle.dump(train_log, open( os.path.join(exp_dir,'log.pkl'), 'wb'))
        # plotting(exp_dir)

    log.close()


if __name__ == '__main__':
    main()