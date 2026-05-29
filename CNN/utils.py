import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import models

from torch.utils.data import DataLoader, Subset
from torchvision import models as models_imagenet

# IMAGE_SIZE = 224
IMAGE_SIZE = 32

def get_dataset(args):
    if args.dataset == 'CIFAR-10':
        print ('CIFAR-10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/', train=True, transform=transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/', train=False, transform=transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'CIFAR-100':
        print ('CIFAR-100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    elif args.dataset == 'MNIST':
        print('MNIST dataset!')
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_dataset = torchvision.datasets.MNIST(
            root="./data/", train=True, download=True, transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ])
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers, pin_memory=True)

        test_dataset = torchvision.datasets.MNIST(
            root="./data/", train=False, download=True, transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ])
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    return train_loader, test_loader


def get_model(args):
    print('Model: {}'.format(args.model))

    if args.dataset == 'CIFAR-10' or args.dataset == 'MNIST':
        num_classes = 10
    elif args.dataset == 'CIFAR-100':
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model_fn = getattr(models, args.model)

    model = model_fn(num_classes=num_classes)

    return model
