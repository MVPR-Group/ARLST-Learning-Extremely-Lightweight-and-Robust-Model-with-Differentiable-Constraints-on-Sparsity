import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class CIFAR10:
    """ 
        CIFAR-10 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[], std=[]
            # mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
            : int(self.args.data_fraction * len(trainset))
        ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader

class CIFAR10_224:
    """
        CIFAR-10 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        self.tr_train = [
            transforms.Resize((224, 224)),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.Resize((224, 224)),
                        transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )


        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


class CIFAR100:
    """ 
        CIFAR-100 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
        )

        self.tr_train = [
            #transforms.Resize((224,224)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [#transforms.Resize((224,224)),
                          transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
            : int(self.args.data_fraction * len(trainset))
        ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


class CIFAR100_224:
    """
        CIFAR-100 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
        )

        self.tr_train = [
            transforms.Resize((224,224)),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.Resize((224,224)),
                          transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
            : int(self.args.data_fraction * len(trainset))
        ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


class CIFAR10_twosize:
    """
        CIFAR-10 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        self.tr_train_224 = [
            transforms.Resize((224, 224)),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test_224 = [transforms.Resize((224, 224)),transforms.ToTensor()]

        self.tr_train_32 = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test_32 = [  transforms.Resize(32),transforms.ToTensor()]

        if normalize:
            self.tr_train_224.append(self.norm_layer)
            self.tr_test_224.append(self.norm_layer)
            self.tr_train_32.append(self.norm_layer)
            self.tr_test_32.append(self.norm_layer)

        self.tr_train_224 = transforms.Compose(self.tr_train_224)
        self.tr_train_32 = transforms.Compose(self.tr_train_32)
        self.tr_test_224 = transforms.Compose(self.tr_test_224)
        self.tr_test_32 = transforms.Compose(self.tr_test_32)

    def data_loaders(self, **kwargs):
        trainset_224 = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train_224,
        )
        trainset_32 = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train_32,
        )


        train_loader_224 = DataLoader(
            trainset_224,
            batch_size=self.args.batch_size,
            shuffle = False,
            **kwargs,
        )
        train_loader_32 = DataLoader(
            trainset_32,
            batch_size=self.args.batch_size,
            shuffle=False,
            **kwargs,
        )
        testset_224 = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test_224,
        )
        testset_32 = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test_32,
        )
        test_loader_224 = DataLoader(
            testset_224, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )
        test_loader_32 = DataLoader(
            testset_32, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader_224.dataset)} images, Test loader: {len(test_loader_224.dataset)} images"
        )
        print(
            f"Traing loader: {len(train_loader_32.dataset)} images, Test loader: {len(test_loader_32.dataset)} images"
        )
        return train_loader_224, test_loader_224, train_loader_32, test_loader_32