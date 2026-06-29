import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

############################################################
# 随机种子
############################################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


############################################################
# Gaussian Blur
############################################################

class GaussianBlur(object):
    """
    SimCLR使用的GaussianBlur
    CIFAR上作用较弱，但保留论文结构
    """

    def __init__(self, kernel_size=3):

        self.blur = transforms.GaussianBlur(
            kernel_size=kernel_size,
            sigma=(0.1,2.0)
        )

    def __call__(self,x):

        return self.blur(x)


############################################################
# TwoCrop
############################################################

class TwoCropTransform:

    """
    同一图片生成两个随机增强视图
    """

    def __init__(self, transform):

        self.transform = transform

    def __call__(self,x):

        return self.transform(x),self.transform(x)


############################################################
# SimCLR数据增强
############################################################

def get_simclr_transform(size=32):

    return transforms.Compose([

        transforms.RandomResizedCrop(
            size=size,
            scale=(0.2,1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ],
            p=0.8
        ),

        transforms.RandomGrayscale(p=0.2),

        GaussianBlur(kernel_size=3),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914,0.4822,0.4465),
            std=(0.2023,0.1994,0.2010)
        )
    ])


############################################################
# Linear Probe增强
############################################################

def get_linear_transform():

    return transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2023,0.1994,0.2010)
        )
    ])


############################################################
# Test增强
############################################################

def get_test_transform():

    return transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2023,0.1994,0.2010)
        )
    ])


############################################################
# SimCLR预训练Loader
############################################################

def get_pretrain_loader(
        train_size=3000,
        batch_size=64,
        seed=42):

    set_seed(seed)

    transform = TwoCropTransform(
        get_simclr_transform()
    )

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    indices = np.random.permutation(len(dataset))

    indices = indices[:train_size]

    subset = Subset(dataset,indices)

    loader = DataLoader(

        subset,

        batch_size=batch_size,

        shuffle=True,

        drop_last=True,

        pin_memory=True,

        num_workers=2 if torch.cuda.is_available() else 0

    )

    return loader,indices


############################################################
# Linear Probe Loader
############################################################

def get_linear_loader(
        indices,
        batch_size=64):

    dataset = datasets.CIFAR10(

        root="data",

        train=True,

        download=True,

        transform=get_linear_transform()

    )

    subset = Subset(dataset,indices)

    loader = DataLoader(

        subset,

        batch_size=batch_size,

        shuffle=True,

        pin_memory=True,

        num_workers=2 if torch.cuda.is_available() else 0

    )

    return loader


############################################################
# Test Loader
############################################################

def get_test_loader(batch_size=64):

    dataset = datasets.CIFAR10(

        root="data",

        train=False,

        download=True,

        transform=get_test_transform()

    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        pin_memory=True,

        num_workers=2 if torch.cuda.is_available() else 0

    )

    return loader