import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class Yale:
    """ 
        imagenet dataset.
    """

    def __init__(self, args, normalize=True):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.tr_train = [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.ImageFolder(
            root = "/home/huangyanhui/hydra-master/datasets/Yale/train", transform=self.tr_train
        )
        testset = datasets.ImageFolder(
            root = "/home/huangyanhui/hydra-master/datasets/Yale/val",transform=self.tr_test
        )

        train_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=False,
            **kwargs,
        )

        test_loader = DataLoader(
            testset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            **kwargs,
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader
