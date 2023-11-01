import os
import torch
import numpy as np

from pathlib import Path
from mpire import WorkerPool
from typing import List, Dict
from collections import defaultdict

from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model


ROOT = Path(__file__).parents[0]

data_file = dict(
    cifar100=ROOT/"CIFAR100_fine_labels.txt"
)

attr_file = dict(
    size=ROOT/'size.txt',
    color=ROOT/'colors.txt'
)


class GenerateTextFeats:
    def __init__(self, prompt, data, attr, attr_adj, njob=1):
        self.njob = njob
        self.batch = 100
        self.prompt = prompt
        self.attr_adj = attr_adj
        self.aug_texts = defaultdict(list)

        with open(attr_file[attr], "r", encoding='utf-8-sig') as f:
            self.attribute = [l.rstrip() for l in f.readlines()]

        with open (data_file[data], "r", encoding='utf-8-sig') as f:
            self.classes = [l.rstrip() for l in f.readlines()]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')
        self.model.to(self.device)
        self.model.eval()

    def make_prompts(self) -> Dict[str, List[str]]:
        print("Prepare prompts...")
        if len(self.classes) > len(self.attribute):
            pool_func = self.by_classes
            chunks_list = self.classes
        else:
            pool_func = self.by_attributes
            chunks_list = self.attribute

        with WorkerPool(self.njob) as pool:
            sub_results = pool.map(pool_func,
                                 chunks_list, iterable_len=len(chunks_list),
                                 n_splits=self.njob, progress_bar=True)

            if isinstance(sub_results[0], dict):
                for sub in sub_results:
                    self.aug_texts.update(sub)
            else:
                for i, cls in enumerate(self.classes):
                    self.aug_texts[cls] = [f'{self.prompt} {cls}']
                    self.aug_texts[cls].extend([
                        sub[i] for sub in sub_results
                    ])

        return self.aug_texts

    def by_classes(self, cls: str) -> Dict[str, List[str]]:
        sub_dict = {f'{cls}': [f'{self.prompt} {cls}']}
        for attr in self.attribute:
            sub_dict[cls].append(
                f'{self.prompt} {attr.lower()} {self.attr_adj} {cls}'
            )
        return sub_dict

    def by_attributes(self, attr: str) -> List[List[str]]:
        sub_list = []
        for cls in self.classes:
            sub_list.append(
                f'{self.prompt} {attr.lower()} {self.attr_adj} {cls}'
            )
        return sub_list

    def chunk_list(self, flat_list: list, n: int) -> List[list]:
        return [sub.tolist() for sub in np.array_split(flat_list, n)]

    def __call__(self, prompts: Dict[str, List[str]]):
        feats = dict()

        with torch.no_grad():

            for cls in self.classes:
                prompt = prompts[cls]
                tokens = []
                for p in prompt:
                    token = self.tokenizer.tokenize(p)
                    token = torch.tensor(self.tokenizer.convert_tokens_to_ids(token)).to(self.device)
                    tokens.append(token)
                
                s = 0
                base_feat, _ = self.model(tokens[0].reshape(1, -1)) 
                base_feat = torch.mean(base_feat, dim=1).reshape(1, 1, -1)
                tokens = tokens[1:]

                diff_feats = []
                while s < len(tokens):
                    txt_inps = tokens[s:s+self.batch]
                    prompt = prompt[s:s+self.batch]
                    feat = []
                    for txt_inp in txt_inps:
                        feat_, _ = self.model(txt_inp.reshape(1, -1))
                        feat_ = torch.mean(feat_, dim=1)
                        feat.append(feat_)
                    feat = torch.cat(feat)

                    diff_feat = base_feat - feat
                    diff_feats.append(diff_feat)
                    s += self.batch

                diff_feats = torch.cat(diff_feats, dim=0)[0]
                diff_feats = torch.cat([base_feat[0], diff_feats], dim=0)
                feats[cls] = dict(
                    zip(prompt, diff_feats.cpu().tolist())
                )

        return feats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--njob', default=2, type=int)
    parser.add_argument('--prompt', default='a photo of', type=str)
    parser.add_argument('--data', default='cifar100', type=str)
    parser.add_argument('--attr', default='size', type=str)
    parser.add_argument('--adj', default='sized', type=str)

    args = parser.parse_args()


    generator = GenerateTextFeats(args.prompt, args.data, args.attr, args.adj, args.njob)

    prompts = generator.make_prompts()
    gen_feats = generator(prompts)

    prompt='_'.join(args.prompt.split())
    save_dir = ROOT/'gpt2_feats'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir / f"{args.data}-{prompt}-{args.attr}-{args.adj}.npy"
    np.save(f'{save_dir.as_posix()}', gen_feats, allow_pickle=True)
