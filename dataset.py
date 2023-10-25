import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import numpy as np

class LabeledCIFAR10(CIFAR10):
    def __init__(self, root, train, download, indices, weak_augment):
        super().__init__(root, train, download=download)
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]
        self.weak_augment = weak_augment()

    def __getitem__(self, idx):
        return self.weak_augment(self.data[idx]), self.targets[idx]


class UnlabeledCIFAR10(CIFAR10):
    def __init__(self, root, train, downlaod, weak_augment, strong_augment):
        super().__init__(root, train, download=downlaod)
        self.weak_augment = weak_augment()
        self.strong_augment = strong_augment()

    def __getitem__(self, idx):
        weak_img = self.weak_augment(self.data[idx])
        strong_img = self.strong_augment(self.data[idx])

        # the targets is returned but not used
        return weak_img, strong_img, self.targets[idx] 
        


def weak_augment():
    return transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])


def strong_augment():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.2),    # Random vertical flip
            transforms.RandomRotation(degrees=30),    # Random rotation up to 15 degrees
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective distortion
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])


def to_tensor():
    return transforms.Compose([
        transforms.ToTensor(), 
    ])

def transform():
    return transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes).type(torch.float32) 

def get_full_dataset(batch_size, num_classes=10):
    train_set = CIFAR10(root='./data', train=True, download=True, transform=weak_augment())
    train_set.targets = one_hot_encode(torch.tensor(train_set.targets), num_classes=num_classes)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def get_subset(batch_size, labels_per_class, num_classes):
    train_set = CIFAR10(root='./data', train=True, download=True, transform=weak_augment())

    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform())
    labels_count = [0 for _ in range(num_classes)]

    labeled_indices = []
    for idx, label in enumerate(train_set.targets):
        if labels_count[label] < labels_per_class:
            labels_count[label] += 1
            labeled_indices.append(idx)

    train_set.targets = one_hot_encode(torch.tensor(train_set.targets), num_classes=num_classes)
    train_set = Subset(train_set, labeled_indices)
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_set, test_set


def get_dataset(labeled_batch_size, unlabeled_batch_size, labels_per_class, num_classes):
    train_set = CIFAR10(root='./data', train=True, download=True)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform())

    labels_count = [0 for _ in range(num_classes)]
    labeled_indices = []

    for idx, label in enumerate(train_set.targets):
        if labels_count[label] < labels_per_class:
            labels_count[label] += 1
            labeled_indices.append(idx)



    labeled_set = LabeledCIFAR10(
        root='./data',
        train=True,
        indices=labeled_indices,
        download=True,
        weak_augment=weak_augment
    )
    labeled_set = DataLoader(labeled_set, batch_size=labeled_batch_size, shuffle=True, num_workers=12, pin_memory=True)


    unlabeled_set = UnlabeledCIFAR10(
        root='./data',
        train=True,
        downlaod=True,
        weak_augment=weak_augment, 
        strong_augment=strong_augment
    )
    unlabeled_set = DataLoader(unlabeled_set, batch_size=unlabeled_batch_size, shuffle=True, num_workers=12, pin_memory=True)


    test_set = DataLoader(test_set, batch_size=labeled_batch_size, shuffle=False, num_workers=8)

    return labeled_set, unlabeled_set, test_set

if __name__ == '__main__':
    get_dataset(64, 64 * 7, 10, 10)