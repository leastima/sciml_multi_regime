import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import csv

from src.pyhessian.hessian import hessian
from src.visualize import *
from utils import *


def predict(inputs, model):
    return model(inputs)


def evaluate(model, dataloader, device):
    """Evaluate accuracy on given dataloader."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # -------------------------------
    # Argument parsing
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet18')
    # parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the loss landscape.')
    parser.add_argument('--type', type=str, default='3D',
                        help='Type of the loss landscape.')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the model.')
    args = parser.parse_args()

    # -------------------------------
    # Environment and parameters
    # -------------------------------
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    density_path = f'./spectral_density/{args.dataset}/{args.batch_size}'
    results_path = f'./results/{args.dataset}/{args.batch_size}'
    os.makedirs(results_path, exist_ok=True)

    # -------------------------------
    # Dataset and model
    # -------------------------------
    train_loader, test_loader = get_dataset(args)

    if args.dataset == 'MNIST':
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        model.maxpool = nn.Identity()
        model.linear = nn.Linear(model.linear.in_features, 10)

        def mnist_forward(x):
            out = F.relu(model.bn1(model.conv1(x)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))        
            out = torch.flatten(out, 1)     
            out = model.linear(out)
            return out
        
        model.forward = mnist_forward
    
    elif args.dataset == "CIFAR-10":
        model = resnet18(pretrained=args.pretrain)

        # 修改输入层，适配 CIFAR-10 (32x32)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # 去掉 ImageNet 用的大尺寸下采样
        model.maxpool = nn.Identity()

        # CIFAR-10: 10 classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    else:
        model = get_model(args)

    model = model.to(device)

    # -------------------------------
    # Loss function & optimizer
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.001
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0
    )

    # -------------------------------
    # Prepare log file
    # -------------------------------
    log_file = os.path.join(results_path, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "test_acc", "hessian_top_eigen"
        ])

    # -------------------------------
    # Hessian spectral density (before training)
    # -------------------------------
    inputs, labels = next(iter(train_loader))
    data = (inputs.to(device), labels.to(device))

    model.eval()
    print("Computing initial Hessian spectral density...")
    comp_hessian = hessian(model, criterion, data)
    eigenval, _ = comp_hessian.eigenvalues(top_n=1)
    top_eig = float(eigenval[0])
    print(f"Initial top eigenvalue of Hessian: {top_eig:.6f}")
    os.makedirs(density_path, exist_ok=True)
    lam_before, dens_before = comp_hessian.density(iter=100, n_v=10)
    np.savez(os.path.join(density_path, 'before_training.npz'),
             eigen=lam_before, weight=dens_before)

    # Record before-training metrics
    train_loss, train_acc = evaluate(model, train_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, train_loss, train_acc, test_acc, top_eig])

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(epochs):
        model.train()
        running_loss, total, correct = 0.0, 0, 0

        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        avg_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Evaluate on test set
        _, test_acc = evaluate(model, test_loader, device)

        # Compute Hessian top eigenvalue after this epoch
        model.eval()
        comp_hessian = hessian(model, criterion, data)
        eigenval, _ = comp_hessian.eigenvalues(top_n=1)
        top_eig = float(eigenval[0])

        if (epoch + 1) % 50 == 0 and (epoch + 1) != 200:
            print("Computing spectral density...")
            lams, dens = comp_hessian.density(iter=100, n_v=10)
            np.savez(
                os.path.join(density_path, f'training_{epoch + 1}epoch.npz'),
                eigen=lams, weight=dens
            )

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, "
              f"Top Eigen: {top_eig:.6f}")

        # Record in log file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, train_acc, test_acc, top_eig])

    # -------------------------------
    # Save final model
    # -------------------------------
    if args.save_model:
        os.makedirs('./saved_models', exist_ok=True)
        torch.save(model.state_dict(), './saved_models/resnet-18.pt')

    # -------------------------------
    # Save final Hessian spectral density
    # -------------------------------
    comp_hessian = hessian(model, criterion, data)
    print("Computing final Hessian spectral density...")
    lam_after, dens_after = comp_hessian.density(iter=100, n_v=10)
    np.savez(os.path.join(density_path, 'after_training.npz'),
             eigen=lam_after, weight=dens_after)

    print(f"Training log saved to {log_file}")

    if args.visualize:
        model.eval()
        eigenvalue, eigenvector = comp_hessian.eigenvalues(top_n=2)
        d1 = eigenvector[0]
        d2 = eigenvector[1]

        inputs, labels = next(iter(train_loader))

        plot_loss_landscape_3d(
            model=model,
            loss_func=criterion,
            inputs=inputs,
            labels=labels,
            d1=d1,
            d2=d2,
            alpha_range=(-0.5, 0.5),
            beta_range=(-0.5, 0.5),
            grid=101,      
            log_scale=True,
            device='cuda',
            elev=30,
            azim=135,
            save_path=f'./loss-landscape/{args.dataset}/{args.batch_size}',
            train_loader=train_loader
        )



if __name__ == '__main__':
    main()
