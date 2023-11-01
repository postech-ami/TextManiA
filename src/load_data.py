'''
Created on 21 Nov 2017


'''
import torch
import os
from torchvision import datasets, transforms
from affine_transforms import Rotation, Zoom
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100, ImbalanceCIFAR100DataLoader, ImbalanceCIFAR10DataLoader

def per_image_standardization(x):
    y = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    mean = y.mean(dim=1, keepdim = True).expand_as(y)    
    std = y.std(dim=1, keepdim = True).expand_as(y)      
    adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.cuda.FloatTensor([x.shape[1]*x.shape[2]*x.shape[3]])))    
    y = (y- mean)/ adjusted_std
    standarized_input =  y.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])  
    return standarized_input  

def load_mnist(data_aug, batch_size, test_batch_size,cuda, data_target_dir):

    if data_aug == 1:
        hw_size = 24
        transform_train = transforms.Compose([
                            transforms.RandomCrop(hw_size),                
                            transforms.ToTensor(),
                            Rotation(15),                                            
                            Zoom((0.85, 1.15)),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.CenterCrop(hw_size),                       
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    else:
        hw_size = 28
        transform_train = transforms.Compose([
                            transforms.ToTensor(),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}       
    
    
                
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader    


def load_data(data_aug, batch_size,workers,dataset, data_target_dir):
    
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                             [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
                                             [ transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    # elif dataset == 'cifar100-lt':
    #     train_data = IMBALANCECIFAR100(data_target_dir, train=True, download=True, transform=train_transform)
    #     test_data = IMBALANCECIFAR100(data_target_dir, train= False, download=True, transform=test_transform)
    # elif dataset == 'cifar10-lt':
    #     train_data = IMBALANCECIFAR10(data_target_dir, train=True, download=True, transform=train_transform)
    #     test_data = IMBALANCECIFAR10(data_target_dir, train= False, download=True, transform=test_transform)
    elif dataset == 'cifar100':    
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'stl10':
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        # assert False, 'Do not finish imagenet code'
        pass
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    
    return train_loader, test_loader, num_classes


def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500, imbalance_factor=100, class_aware=False):
    ## copied from GibbsNet_pytorch/load.py
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
        
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100' or 'cifar100-lt':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'tiny-imagenet-200':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'mnist':
        pass 
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        print ('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                             [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif dataset == 'mnist':
            hw_size = 24
            train_transform = transforms.Compose([
                                transforms.RandomCrop(hw_size),                
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.CenterCrop(hw_size),                       
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
        elif dataset == 'tiny-imagenet-200':
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(64, padding=4),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:    
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(32, padding=2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        if dataset == 'mnist':
            hw_size = 28
            train_transform = transforms.Compose([
                                transforms.ToTensor(),       
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
                
        else:   
            train_transform = transforms.Compose(
                                                 [transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar10-lt':
        # train_data = IMBALANCECIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        # test_data = IMBALANCECIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100-lt':
        # train_data = IMBALANCECIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        # test_data = IMBALANCECIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'imagenet-lt':
        num_classes = 1000
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'stl10':
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    
    elif dataset == 'tiny-imagenet-200':
        if labels_per_class == 50:
            train_root = os.path.join(data_target_dir, 'train0.1') 
        else:
            train_root = os.path.join(data_target_dir, 'train')
       
        validation_root = os.path.join(data_target_dir, 'val/images')
        test_data = datasets.ImageFolder(validation_root,transform=test_transform)
        num_classes = 200
        
    elif dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(data_target_dir, 'train')
        valdir = os.path.join(data_target_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_data = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_data = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

        
    n_labels = num_classes
    
    def get_sampler(labels, n=None, n_valid= None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        
        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled
    
    
    # Dataloaders for MNIST
    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
    elif dataset == 'mnist':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
    elif dataset == 'tiny-imagenet-200':
        pass
    elif dataset == 'cifar100-lt':
        pass
    elif dataset == 'cifar100':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class, valid_labels_per_class)
    else: 
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class, valid_labels_per_class)

    if dataset == 'tiny-imagenet-200':
        labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        validation = None
        unlabelled = None
        test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    elif dataset == "cifar100-lt":
        labelled = ImbalanceCIFAR100DataLoader(data_dir=data_target_dir,batch_size=batch_size,num_workers=workers,training=True,retain_epoch_size=True,imb_factor = imbalance_factor)
        validation = None
        unlabelled = None
        test = ImbalanceCIFAR100DataLoader(data_dir=data_target_dir,batch_size=batch_size,num_workers=workers,training=False,retain_epoch_size=True,imb_factor = imbalance_factor)
    elif dataset == "cifar10-lt":
        labelled = ImbalanceCIFAR10DataLoader(data_dir=data_target_dir,batch_size=batch_size,num_workers=workers,training=True,retain_epoch_size=True,imb_factor = imbalance_factor)
        validation = None
        unlabelled = None
        test = ImbalanceCIFAR10DataLoader(data_dir=data_target_dir,batch_size=batch_size,num_workers=workers,training=False,retain_epoch_size=True,imb_factor = imbalance_factor)
    elif dataset == "imagenet":
        labelled = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

        unlabelled = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
    else:
        labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        validation = None
        unlabelled = None
        test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes

def load_data_subset_unpre(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    ## loads the data without any preprocessing##
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    
    """
    def per_image_standarize(x):
        mean = x.mean()
        std = x.std()
        adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.FloatTensor([x.shape[0]*x.shape[1]*x.shape[2]])))
        standarized_input = (x- mean)/ adjusted_std
        return standarized_input
    """   
    if data_aug==1:
        print ('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                            [transforms.RandomCrop(32, padding=2),
                                            transforms.ToTensor(), 
                                            transforms.Lambda(lambda x : x.mul(255))
                                            ])
        else:    
            train_transform = transforms.Compose(
                                                [transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, padding=2),
                                                transforms.ToTensor(), 
                                                transforms.Lambda(lambda x : x.mul(255))
                                                ])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x : x.mul(255))])
    else:
        print ('no data aug')
        train_transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x : x.mul(255))
                                            ])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x : x.mul(255))])
    
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    #print ('svhn', train_data.labels.shape)
    elif dataset == 'stl10':
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        # assert False, 'Do not finish imagenet code'
        pass
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)
        
    n_labels = num_classes
    
    def get_sampler(labels, n=None, n_valid= None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        #print type(labels)
        #print (n_valid)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        
        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])
        #print (indices_train.shape)
        #print (indices_valid.shape)
        #print (indices_unlabelled.shape)
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled
    
    
    # Dataloaders for MNIST
    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
    elif dataset == 'mnist':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
    else: 
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class)
    
    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,shuffle=False,  num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes

