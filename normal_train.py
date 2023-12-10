# Theoretically Principled Trade-off between Robustness and Accuracy

from __future__ import print_function

import argparse
import os
import time
import warnings

import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from utils import Logger, random_seed, save_checkpoint

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PyTorch CIFAR No Defense")
parser.add_argument(
    "--arch", type=str, default="ResNet18", choices=["WideResNet", "ResNet18"]
)
# data
parser.add_argument(
    "--data", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"]
)
parser.add_argument(
    "--data-path", type=str, default="~/datasets/", help="where is the dataset CIFAR-10"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 128)",
)
# traning setting
parser.add_argument(
    "--epochs", type=int, default=200, metavar="N", help="number of epochs to train"
)
parser.add_argument("--weight-decay", "--wd", default=5e-4, type=float, metavar="W")
parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate")
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

# resume
parser.add_argument(
    "--start-epoch", type=int, default=1, metavar="N", help="retrain from which epoch"
)
parser.add_argument(
    "--resume_path", default="", type=str, help="directory of model for retraining"
)
# save checkpoint
parser.add_argument(
    "--result-dir",
    default="results/Normal_Train",
    help="directory of model for saving checkpoint",
)
parser.add_argument(
    "--save-freq", "-s", default=1, type=int, metavar="N", help="save frequency"
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

args = parser.parse_args()

if args.data == "CIFAR100":
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10


if args.seed is not None:
    random_seed(args.seed)

args_path = (
    "epoch"
    + str(args.epochs)
    + "_bs"
    + str(args.batch_size)
    + "_lr"
    + str(args.lr)
    + "_wd"
    + str(args.weight_decay)
)

checkpoint_path = os.path.join(
    args.result_dir, args.data, args.arch, args_path, "checkpoints"
)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

writer = SummaryWriter(
    os.path.join(args.result_dir, args.data, args.arch, args_path, "tensorboard_logs")
)

logger = Logger(
    os.path.join(args.result_dir, args.data, args.arch, args_path, "output.log")
)

best_nature_acc = 0

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

trainset = getattr(datasets, args.data)(
    root=args.data_path, train=True, download=False, transform=transform_train
)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=False, transform=transform_test
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, **kwargs
)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(batch_idx)
        optimizer.zero_grad()

        outputs = model(data)

        loss = nn.CrossEntropyLoss()(outputs, target)

        loss.backward()

        optimizer.step()


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    global best_nature_acc

    logger.info(args)

    # load model
    model = getattr(models, args.arch)(num_classes=NUM_CLASSES)
    # dataparallel
    model = nn.DataParallel(model).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.start_epoch > 1:
        logger.info("Retrain from epoch %d", (args.start_epoch))
        state_dict = torch.load(args.resume_path, map_location=device)
        optimizer.load_state_dict(state_dict["opt_state_dict"])
        model.load_state_dict(state_dict["model_state_dict"])

    logger.info("Epoch \t Time \t Train Loss \t Train ACC \t Test Loss \t Test ACC")
    for epoch in range(args.start_epoch, args.epochs + 1):
        # train
        torch.cuda.synchronize()
        start_time = time.time()

        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

        torch.cuda.synchronize()
        epoch_time = time.time() - start_time

        # eval
        train_loss, train_acc = eval_train(model, device, train_loader)
        test_loss, test_acc = eval_test(model, device, test_loader)

        logger.info(
            "%d\t %.4f \t %.4f \t\t %.4f \t\t %.4f \t %.4f",
            epoch,
            epoch_time,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        )

        writer.add_scalar("Train/Loss", float(train_loss), epoch)
        writer.add_scalar("Train/Acc", float(train_acc), epoch)
        writer.add_scalar("Test/Loss", float(test_loss), epoch)
        writer.add_scalar("Test/Acc", float(test_acc), epoch)

        # Save checkpoint
        is_best = test_acc > best_nature_acc
        best_nature_acc = max(test_acc, best_nature_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "nature_acc": float(test_acc),
            },
            epoch,
            is_best,
            "nature",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

    logger.info("Best Nature ACC %.4f", best_nature_acc)

    writer.close()


if __name__ == "__main__":
    main()