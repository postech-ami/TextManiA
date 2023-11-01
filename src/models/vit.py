import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.autograd import Variable
from utils import to_one_hot, mixup_process, get_lambda, mixup_data
from load_data import per_image_standardization

import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class VIT(nn.Module):
    def __init__(self, model_name, num_classes, image_resolution, embedding_dim, mode = "interpolation", per_img_std=False):
        super(VIT, self).__init__()
        self.per_img_std = per_img_std
        self.mode = mode
        self.num_classes = num_classes
        # self.embedding_dim = embedding_dim
        if num_classes == 200: # dataset is imagenet-tiny
            self.image_size = 64 # train dataset image resolution
        else:
            self.image_size = 32

        self.image_resolution = image_resolution #model's input image resolution
        self.padding_size = (self.image_resolution-self.image_size)//2
        self.diff_projection = nn.Linear(512, embedding_dim)
        model = timm.create_model(model_name, pretrained=True)


        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.no_embed_class = model.no_embed_class

        self.num_prefix_tokens = model.num_prefix_tokens
        self.global_pool = model.global_pool
        self.patch_embed = model.patch_embed
        self._pos_embed = model._pos_embed
        self. norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm = model.norm
        self.fc_norm = model.fc_norm
        self.head = nn.Linear(embedding_dim, num_classes)


    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)




    def forward(self, x, target=None,
                mixup=False, mixup_hidden=False, mixup_alpha=None,
                diff_vec=None, aug_target=None, cutmix=False,
                scale=0.1, max_scale=1.0
    ):
        if self.per_img_std:
            x = per_image_standardization(x)

        if mixup_hidden:
            layer_mix = random.randint(0, 2)
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


        if self.mode == "interpolation":
            out = F.interpolate(x, size=self.image_resolution)
        elif self.mode == "pad":
            out = F.pad(x, (self.padding_size,self.padding_size,self.padding_size,self.padding_size))


        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam, cutmix=cutmix)

        out = self.patch_embed(out)
        out = self._pos_embed(out)
        out = self.norm_pre(out)
        out = self.blocks(out)
        out = self.norm(out)

        if self.global_pool:
            if self.global_pool == 'avg':
                out = out[:, self.num_prefix_tokens:].mean(dim=1)
            else:
                out = out[:, 0]




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
            out = torch.cat([out, aug_feats], dim=0)
            out = self.fc_norm(out)
            out = self.head(out)
            ori_feats = out[:len(x)]
            aug_feats = out[len(x):]


            if target is not None and aug_targets is not None:
                aug_targets = torch.cat(aug_targets, dim=0)
                return ori_feats, aug_feats, target_reweighted, aug_targets
            else:
                return ori_feats, aug_feats
        else:

            out = self.fc_norm(out)
            out = self.head(out)
            if target is not None:
                return out, target_reweighted

            else:
                return out




def vit_tiny_patch16_224(num_classes=100, dropout = False, per_img_std = False, stride=1, mode='interpolation'):
    model = VIT(model_name = "vit_tiny_patch16_224", num_classes = num_classes, image_resolution = 224, embedding_dim = 192, mode=mode)
    return model

def vit_small_patch16_224(num_classes=100, dropout = False, per_img_std = False, stride=1, mode = 'interpolation'):
    model = VIT(model_name = "vit_small_patch16_224", num_classes = num_classes, image_resolution = 224, embedding_dim = 384, mode=mode)
    return model

def vit_base_patch16_224(num_classes=100, dropout = False, per_img_std = False, stride=1, mode = 'interpolation'):
    model = VIT(model_name = "vit_base_patch16_224", num_classes = num_classes, image_resolution = 224, embedding_dim = 768, mode=mode)
    return model