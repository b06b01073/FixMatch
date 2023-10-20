import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

def weak_augment():
    return transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
    ])

def transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes).type(torch.float32) 

def get_full_dataset(batch_size, num_classes=10):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=weak_augment())
    train_set.targets = one_hot_encode(torch.tensor(train_set.targets), num_classes=num_classes)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform())
    test_set.targets = one_hot_encode(torch.tensor(test_set.targets), num_classes=num_classes)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def get_dataset(labeled_batch_size, unlabeled_batch_size, labels_per_class, num_classes):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    labels_count = [0 for _ in range(num_classes)]
    labeled_indices = []
    unlabeled_indices = []

    for idx, label in enumerate(train_set.targets):
        if labels_count[label] < labels_per_class:
            labels_count[label] += 1
            labeled_indices.append(idx)
        else:
            unlabeled_indices.append(idx)

    labeled_set = Subset(train_set, labeled_indices)
    labeled_set.dataset.transform = weak_augment()
    labeled_set = DataLoader(labeled_set, batch_size=labeled_batch_size, shuffle=True, num_workers=8)

    unlabeled_set = Subset(train_set, unlabeled_indices)
    unlabeled_set = DataLoader(unlabeled_set, batch_size=unlabeled_batch_size, shuffle=True)

    test_set = DataLoader(test_set, batch_size=labeled_batch_size, shuffle=False, num_workers=8)

    return labeled_set, unlabeled_set, test_set

if __name__ == '__main__':
    get_dataset(64, 64 * 7, 10, 10)