import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union, Optional

from models.utils import to_one_hot


ROOT = Path(__file__).parents[1]


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


class TextManiA(nn.Module):
    def __init__(self, args, prompts, attr, adj,
                 max_sample=-1, nsample=False, scale=1.0):
        super(TextManiA, self).__init__()

        self.adj = adj  # ['colored', 'sized']
        self.attr = attr  # ['color', 'size']
        self.prompts = prompts if isinstance(prompts, list) else [prompts]
        self.max_nsample = max_sample
        self.nsample = nsample
        self.scale = scale

        self.prompts = [
            "_".join(prompt.split()) for prompt in self.prompts
        ]

        self.base_feat: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.attr_dict: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(dict)
        self.tail=False

        if args.dataset == 'cifar100-lt':
            self.tail=True
            args.dataset = 'cifar100'
        elif args.dataset == 'cifar10-lt':
            args.dataset = 'cifar10'
            self.tail=True
        else:
            pass

        for prompt in self.prompts:
            if not prompt: continue

            for attr, adj in zip(self.attr, self.adj):
                feat_path = ROOT / 'preprocessing/feats' / f'{args.dataset}-{prompt}-{attr}-{adj}.npy'
                attr_dict = np.load(feat_path.as_posix(), allow_pickle=True).item()

                for cls in attr_dict:
                    feats = torch.tensor(list(attr_dict[cls].values())).float()
                    base_feat = feats[0].reshape(1, -1)
                    attr_feat = feats[1:]

                    if cls in self.base_feat and prompt in self.base_feat[cls]:
                        assert torch.allclose(
                            base_feat,
                            self.base_feat[cls][prompt]
                        )
                    elif cls not in self.base_feat or prompt not in self.base_feat[cls]:
                        self.base_feat[cls][prompt] = base_feat

                    if prompt not in self.attr_dict[cls]:
                        self.attr_dict[cls][prompt] = [attr_feat]
                    else:
                        self.attr_dict[cls][prompt].append(attr_feat)

        del attr_dict
        self.promot_count: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for cls in self.attr_dict:
            for prompt in self.attr_dict[cls]:
                self.attr_dict[cls][prompt] = torch.cat(self.attr_dict[cls][prompt], dim=0)
                prompt_count = torch.tensor([0] * len(self.attr_dict[cls][prompt])).float()
                self.promot_count[cls][prompt] = prompt_count

        self.class_name = list(self.attr_dict.keys())
        self.prompts = list(self.attr_dict[self.class_name[0]].keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = args.num_classes

    def label2class(self, label):
        label_ = np.array(label.cpu())
        return np.array(self.class_name)[label_].tolist()

    def text_aug(self,
                 labels: torch.Tensor,
                 img_feats: Optional[torch.Tensor] = None,
                 projection_layer: Optional[torch.nn.Linear] = None
    ) -> List[torch.Tensor]:
        input_cls = self.label2class(labels)

        base_feat = []
        aug_targets = []
        sampled_feats = []
        for i, inst in enumerate(input_cls):
            if len(self.prompts) > 1:
                prompt_w = torch.tensor([w.sum() for w in self.promot_count[inst].values()])
                prompt_w = 1 - F.normalize(prompt_w, dim=0)
                prompt_i = torch.multinomial(prompt_w, num_samples=1)
                prompt = self.prompts[prompt_i]
            else:
                prompt = self.prompts[0]

            if self.max_nsample == -1:
                max_n = len(self.attr_dict[inst][prompt])
            else:
                max_n = self.max_nsample

            if self.nsample:
                nsample = max_n
            else:
                nsample = random.randint(1, max_n)

            aug_target = torch.tensor([labels[i].item()] * nsample).long().reshape(-1, 1)
            aug_targets.extend(aug_target)
            w = 1 - F.normalize(self.promot_count[inst][prompt], dim=0)
            ids = torch.multinomial(w, num_samples=nsample)
            self.promot_count[inst][prompt][ids] += 1

            sampled_feat = self.attr_dict[inst][prompt][ids]
            if projection_layer is not None:
                sampled_feat = sampled_feat.to(img_feats.device)
                sampled_feat = projection_layer(sampled_feat)

            if img_feats is not None:
                alpha = torch.rand((sampled_feat.size(0), 1)).to(img_feats.device).clamp_min(self.scale)
                sampled_feat = img_feats[i].reshape(1, 1, -1) + (alpha * sampled_feat)[None]
                sampled_feat = sampled_feat[0]
            sampled_feats.append(sampled_feat)
            base_feat.append(self.base_feat[inst][prompt])

        # prompt
        if img_feats is not None:
            diff_feats = torch.cat(sampled_feats, dim=0)
        else:
            diff_feats = sampled_feats
        base_feat = torch.cat(base_feat, dim=0)
        aug_targets = torch.cat(aug_targets).long().to(self.device)
        return [base_feat, diff_feats, aug_targets]

    def __call__(self,
                 label: torch.Tensor,
                 img_feats: Optional[torch.Tensor] = None,
                 projection_layer: Optional[torch.nn.Linear] = None
    ) -> List[torch.Tensor]:
        base_feat, diff_feats, aug_targets = self.text_aug(label, img_feats, projection_layer)
        return [base_feat, diff_feats, aug_targets]