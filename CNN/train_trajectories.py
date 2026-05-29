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
import powerlaw
import random
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.resnet import *
from torch.nn.utils import parameters_to_vector
import csv

from src.pyhessian.hessian import hessian
from src.visualize import *
from utils import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(seed, model, device, lr, batch_size, loss_threshold=1e-2):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data/', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    log_path = f'./logs/lr_{lr}_bs_{batch_size}_seed_{seed}'
    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])

    epochs = 250
    for epoch in range(epochs):
        model.train()
        running_loss, total_samples = 0.0, 0

        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        total_loss = running_loss / total_samples

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, total_loss])
        
        if (epoch + 1) % 10 == 0:
            print(f"Train Loss: {total_loss:.4f}")

        if total_loss <= loss_threshold:
            print(f"Epoch[{epoch + 1}/{epochs}]: train_loss={total_loss:.4f}")
            break

    max_iters = 200
    distance_list = []
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        if i >= max_iters:
            break

        theta_1 = parameters_to_vector(model.parameters())

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        theta_2 = parameters_to_vector(model.parameters())

        distance = torch.norm(theta_1 - theta_2)
        distance_list.append(distance.item())

    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_acc = 100.0 * correct / total

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total

    acc_gap = train_acc - test_acc

    return acc_gap, distance_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = ResNet18()
    model.to(device)

    acc_gap, seq_distance = train(args.seed, model, device, args.lr, args.batch_size, loss_threshold=1e-4)

    seq_observe = []

    for i in range(len(seq_distance)):
        if seq_distance[i] > 1e-12:
            seq_observe.append(1 / seq_distance[i]) 

    seq_observe = np.array(seq_observe)
    fit = powerlaw.Fit(seq_observe)

    result = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "alpha": fit.alpha,
        "acc_gap": acc_gap
    }

    with open("./results/results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()