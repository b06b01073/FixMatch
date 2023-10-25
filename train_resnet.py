from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import ResNet
import dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--labels_per_class', '-l', type=int)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='store_false')
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='model_params')
    parser.add_argument('--file_name', '-f', type=str, default='resnet.pth')

    args = parser.parse_args()
    
    train_set, test_set = dataset.get_full_dataset(args.batch_size)

    model = ResNet(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model.fit(
        train_set, 
        test_set, 
        args.epoch, 
        loss_fn,
        optimizer,
        args.verbose,
        args.save_dir,
        args.file_name
    )

