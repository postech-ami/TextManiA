## https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.autograd import Variable
from utilsback import to_one_hot, mixup_process, get_lambda
from load_data import per_image_standardization
import random


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, per_img_std = False):
        super(ResNet, self).__init__()
        self.per_img_std = per_img_std
        self.in_planes = 64
        self.num_classes = num_classes

        if self.num_classes == 200:
            initial_stride=2
        else:
            initial_stride=1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=initial_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if block.expansion == 1:
            self.diff_projection = None
        else:
            self.diff_projection = nn.Linear(512, 512*block.expansion)
    

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target=None,
                mixup=False, mixup_hidden=False, mixup_alpha=None,
                diff_vec=None, aug_target=None, cutmix=False, noise=None,align_mixup=False,
                scale=0.1, max_scale=1.0, class_aware=False
    ):
        # import pdb; pdb.set_trace()
        if self.per_img_std:
            x = per_image_standardization(x)
        ###
        if mixup_hidden and not align_mixup:
            layer_mix = random.randint(0, 2)
        elif align_mixup:
            layer_mix = random.randint(0, 1)
        elif mixup or cutmix:
            layer_mix = 0
        else:
            layer_mix = None

        out = x

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)

        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)
            target_reweighted = target_reweighted.to(target.device)
        
        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam, cutmix=cutmix, align_mixup=True)

        out = self.conv1(out)
        out = self.layer1(out)

        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer2(out)

        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer3(out)
        if layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer4(out)
        
        if layer_mix == 1 and align_mixup:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam, cutmix=cutmix, align_mixup=align_mixup)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        if diff_vec is not None and self.training:
            aug_feats = []
            aug_targets = []
            aug_out = out.clone()
            for i, diff in enumerate(diff_vec):
                if self.diff_projection is not None:
                    diff = self.diff_projection(diff.to(out.device))
                else:
                    diff = diff.to(out.device)
                # alpha = torch.rand((diff.size(0), 1)).clamp(min=scale, max=max_scale).to(out.device)
                alpha = torch.randn((diff.size(0), 1)).clamp(min=scale).to(out.device)
                aug_feats.append(
                    (aug_out[i].reshape(1, 1, -1) + (alpha * diff)[None])[0]
                )

                if aug_target is not None:
                    aug_targets.append(
                        target_reweighted[i][None] * (1 - alpha) + (to_one_hot(aug_target[i][:, 0].to(out.device), self.num_classes) * alpha)
                    )
            aug_feats = torch.cat(aug_feats, dim=0)
            out = self.linear(
                torch.cat([out, aug_feats], dim=0)
            )

            ori_feats = out[:len(x)]
            aug_feats = out[len(x):]

            if target is not None and aug_targets is not None:
                aug_targets = torch.cat(aug_targets, dim=0)
                return ori_feats, aug_feats, target_reweighted, aug_targets
            else:
                return ori_feats, aug_feats
        elif noise is not None and self.training:
            noise_vec = torch.normal(0,noise,size=out.size()).to(out.device)
            # noise_vec.to(out.device)
            # noise_vec = torch.rand(out.size(), dtype=out.dtype, layout=out.layout, device=out.device)
            # noise_vec= torch.div(noise_vec, noise) # 10 : 0~0.1
            out = out + noise_vec
            out = self.linear(out)
            if target is not None:
                return out, target_reweighted
            else:
                return out
        else:
            out = self.linear(out)
            if target is not None:
                return out, target_reweighted
            else:
                return out

def resnet18(num_classes=10, dropout = False, per_img_std = False, stride=1):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, per_img_std = per_img_std)
    return model


def resnet34(num_classes=10, dropout = False, per_img_std = False, stride=1):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, per_img_std = per_img_std)
    return model


def resnet50(num_classes=10, dropout = False, per_img_std = False, stride=1):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, per_img_std = per_img_std)
    return model


def resnet101(num_classes=10, dropout = False, per_img_std = False, stride=1):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, per_img_std = per_img_std)
    return model


def resnet152(num_classes=10, dropout = False, per_img_std = False, stride=1):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, per_img_std = per_img_std)
    return model