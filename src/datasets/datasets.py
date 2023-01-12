import torch
import random
from torchvision import transforms
from PIL import ImageFilter, Image

from src.datasets.cifar10 import CIFAR10, CIFAR10Corners, CIFAR10Shapes, CIFAR10Digits, CIFAR10Letters

from src.datasets.data_statistics import get_data_mean_and_stdev

DATASET = {
    'cifar10': CIFAR10,
    'cifar10shapes': CIFAR10Shapes,
    'cifar10digits': CIFAR10Digits,
    'cifar10letters': CIFAR10Letters,
    'cifar10_corners': CIFAR10Corners,
}


def zscore_image(img_tensor):
    img_tensor -= img_tensor.mean([-1, -2], keepdim=True)
    img_tensor /= img_tensor.std([-1, -2], keepdim=True)
    return img_tensor

def get_image_datasets(
        dataset_name,
        default_augmentations='none',
        alternate_label=False  # Whether to use e.g. shape type instead of the image label.
    ):
    load_transforms = TRANSFORMS[default_augmentations]
    train_transforms, test_transforms = load_transforms(
        dataset=dataset_name, 
    )
    train_dataset = DATASET[dataset_name](
        train=True,
        image_transforms=train_transforms,
        alternate_label=alternate_label
    )
    val_dataset = DATASET[dataset_name](
        train=False,
        image_transforms=test_transforms,
        alternate_label=alternate_label
    )
    return train_dataset, val_dataset

def get_image_datasets_val(
        dataset_name,
        default_augmentations='none',
        alternate_label=False  # Whether to use e.g. shape type instead of the image label.
    ):
    load_transforms = TRANSFORMS[default_augmentations]
    train_transforms, test_transforms = load_transforms(
        dataset=dataset_name, 
    )
    train_dataset = DATASET[dataset_name](
        train=True,
        image_transforms=train_transforms,
        alternate_label=alternate_label
    )
    val_dataset = DATASET[dataset_name](
        train=False,
        image_transforms=train_transforms,
        alternate_label=alternate_label
    )
    return train_dataset, val_dataset

def load_image_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


def load_default_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        mean, std = get_data_mean_and_stdev(dataset)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return None, None
    
    return train_transforms, test_transforms


def load_default_unnorm_transforms(dataset, **kwargs):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.ToTensor()
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


TRANSFORMS = {
    True: load_default_transforms,
    False: load_image_transforms,
    'all': load_default_transforms,
    'all_unnorm': load_default_unnorm_transforms,
    'none': load_image_transforms, 
}
