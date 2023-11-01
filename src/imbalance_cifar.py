# From: https://github.com/kaidic/LDAM-DRW
import torch
import torchvision
import numpy as np
import random
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image

# imbalance factor : 100 = 1/0.01 -> set imb_factor = 0.01
# imbalance factor : 50 = 1/0.02 -> set imb_factor = 0.02
# imbalance factor : 10 = 1/0.1 -> set imb_factor = 0.1

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=43, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

class LT_Dataset(Dataset):
    def __init__(self,root,txt,transform):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_paths.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        path,label = self.img_paths[index],self.labels[index]
        with open(path,'rb') as f:
            img = Image.open(f).convert('RGB')
            img = self.transform(img)
        return img,label
    
class ImbalanceCIFAR100DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True,imb_factor=100):
        normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32,padding=4) ,
            transforms.RandomHorizontalFlip()   ,
            transforms.RandomRotation(15)       ,
            transforms.ToTensor()               ,
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])
 
        if imb_factor==100:
            self.imb_factor = 0.01
        elif imb_factor==50:
            self.imb_factor = 0.02
        elif imb_factor==10:
            self.imb_factor = 0.1
        else:
            self.imb_factor = 0.01

        if training:
            self.dataset = IMBALANCECIFAR100(data_dir,train=True,download=True,transform=train_trsfm, imb_factor=self.imb_factor)
            self.val_dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)
            self.val_dataset = None

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 100

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : False
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=None)

    def split_validation(self,type='test'):
        return DataLoader(
            dataset     = self.OOD_dataset if type=='OOD' else self.val_dataset ,
            batch_size  = 4096                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )

class ImbalanceCIFAR10DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True,imb_factor=100):
        normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()   ,
            transforms.RandomRotation(15)       ,
            transforms.ToTensor()               ,
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])
        if imb_factor==100:
            self.imb_factor = 0.01
        elif imb_factor==50:
            self.imb_factor = 0.02
        elif imb_factor==10:
            self.imb_factor = 0.1
        else:
            self.imb_factor = 0.01

        if training:
            self.dataset = IMBALANCECIFAR10(data_dir,train=True,download=True,transform=train_trsfm,imb_factor=self.imb_factor)
            self.val_dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_trsfm)
            self.val_dataset = None

        # Uncomment to use OOD datasets
        self.OOD_dataset = None
        # self.OOD_dataset = datasets.SVHN(data_dir,split="test",download=True,transform=test_trsfm)
        # self.OOD_dataset = LT_Dataset('/local_data/ImageNet_LT/ImageNet_LT_open','/local_data/ImageNet_LT/ImageNet_LT_open.txt',train_trsfm)
        # self.OOD_dataset = LT_Dataset('../Places_LT/Places_LT_open','../Places_LT/Places_LT_open.txt',train_trsfm)

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 10

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : False
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=None)

    def split_validation(self,type='test'):
        return DataLoader(
            dataset     = self.OOD_dataset if type=='OOD' else self.val_dataset ,
            batch_size  = 4096                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )