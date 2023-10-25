from argparse import ArgumentParser
import dataset
from model.model import FixMatch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import math


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--tau', '-t', default=0.95, type=float, help='the threshold')
    parser.add_argument('--lamb', '-l', default=1, type=float, help='\lambda_\mu in the paper')
    parser.add_argument('--mu', '-m', default=7, type=int, help='the scale of unlabeled dataset compared to the labeled dataset')
    parser.add_argument('--batch_size', default=48, type=int, help='batch size')
    parser.add_argument('--lr', default=3e-2, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', action='store_false')
    parser.add_argument('--weight_decay', default=5e-4,type=float)
    parser.add_argument('--steps', default=int(1e6), type=int)
    parser.add_argument('--labels_per_class', type=int, default=25)
    parser.add_argument('--save_dir', type=str, default='./model_params')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--file_name', type=str, default='fixmatch.pth')
    parser.add_argument('--pretrained', type=str)

    args = parser.parse_args()

    labeled_set, unlabeled_set, test_set = dataset.get_dataset( 
        args.batch_size, 
        args.batch_size * args.mu, 
        args.labels_per_class, 
        args.num_classes
    )

    
    
    fixmatch = FixMatch(
        labeled_set, 
        unlabeled_set, 
        test_set, 
        args.num_classes,
        args.tau,
        args.lamb,
        device
    )

    if args.pretrained:
        fixmatch.load(args.pretrained)

    optimizer = optim.SGD(
        fixmatch.net.parameters(), 
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10)


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    fixmatch.fit(
        args.steps,
        optimizer,
        scheduler,
        args.verbose,
        args.save_dir,
        args.file_name,
        args.log_interval
    )
