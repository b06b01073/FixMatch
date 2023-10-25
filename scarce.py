from argparse import ArgumentParser
import dataset
from model.model import ResNet
import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=48, type=int, help='batch size')
    parser.add_argument('--lr', default=3e-2, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', action='store_false')
    parser.add_argument('--weight_decay', default=5e-4,type=float)
    parser.add_argument('--labels_per_class', type=int, default=25)
    parser.add_argument('--save_dir', type=str, default='./model_params')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', '-e', default=500, type=int)
    parser.add_argument('--file_name', '-f', default='scarce.pth', type=str)
    parser.add_argument('--test_interval', default=25, type=int)

    args = parser.parse_args()

    train_set, test_set = dataset.get_subset( 
        args.batch_size, 
        args.labels_per_class, 
        args.num_classes
    )

    resnet50 = ResNet(device)

    optimizer = optim.SGD(
        resnet50.net.parameters(), 
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10)

    loss_fn = nn.CrossEntropyLoss()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    resnet50.fit(
        train_set,
        test_set,
        args.epochs,
        loss_fn,
        optimizer,
        scheduler,
        args.verbose,
        args.save_dir,
        args.file_name,
        args.test_interval
    )




