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

parser = argparse.ArgumentParser(description="PyTorch CIFAR TRADES Defense")
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
    "--epochs", type=int, default=76, metavar="N", help="number of epochs to train"
)
parser.add_argument("--weight-decay", "--wd", default=2e-4, type=float, metavar="W")
parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate")
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
# TRADES setting
parser.add_argument(
    "--norm",
    default="l_inf",
    type=str,
    choices=["l_inf", "l_2"],
    help="The threat model",
)
parser.add_argument("--epsilon", default=8.0 / 255, type=eval, help="perturbation")
parser.add_argument("--num-steps", default=10, type=int, help="perturb number of steps")
parser.add_argument(
    "--step-size", default=2.0 / 255, type=eval, help="perturb step size"
)
parser.add_argument(
    "--beta", default=6.0, type=float, help="regularization, i.e., 1/lambda in TRADES"
)
# Eval PGD setting
parser.add_argument("--test-epsilon", default=8.0 / 255, type=eval, help="perturbation")
parser.add_argument(
    "--test-step-size", default=2.0 / 255, type=eval, help="perturb step size"
)
parser.add_argument(
    "--test-num-steps", default=20, type=int, help="perturb number of steps"
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
    default="results/TRADES",
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
    + "_eps"
    + str(args.epsilon)
    + "_norm"
    + str(args.norm)
    + "_beta"
    + str(args.beta)
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
best_robust_acc = 0

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
    root=args.data_path, train=True, download=True, transform=transform_train
)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=True, transform=transform_test
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)


def trades_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size=2.0 / 255,
    epsilon=8.0 / 255,
    perturb_steps=10,
    beta=6.0,
    distance="l_inf",
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == "l_inf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), F.softmax(model(x_natural), dim=1)
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return loss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(
            model=model,
            x_natural=data,
            y=target,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            distance=args.norm,
        )
        loss.backward()
        optimizer.step()


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _pgd_whitebox(
    model, X, y, epsilon=0.031, step_size=0.003, num_steps=20, random=True
):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = (
            torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    return err, err_pgd


def eval_adv_whitebox(
    model, device, test_loader, epsilon, step_size, num_steps, random
):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    count = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        count += X.shape[0]
        err_natural, err_robust = _pgd_whitebox(
            model, X, y, epsilon, step_size, num_steps, random
        )
        robust_err_total += err_robust
        natural_err_total += err_natural

    nature_acc = 1.0 - (natural_err_total / count)
    robust_acc = 1.0 - (robust_err_total / count)

    return nature_acc, robust_acc


def main():
    global best_nature_acc, best_robust_acc

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

    if args.start_epoch > 1:
        logger.info("Retrain from epoch %d", (args.start_epoch))
        state_dict = torch.load(args.resume_path, map_location=device)
        optimizer.load_state_dict(state_dict["opt_state_dict"])
        model.load_state_dict(state_dict["model_state_dict"])

    logger.info("Epoch \t Time \t Test ACC \t Test Robust ACC")
    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        start_time = time.time()
        # train
        train(args, model, device, train_loader, optimizer, epoch)

        # eval
        nature_acc, robust_acc = eval_adv_whitebox(
            model,
            device,
            test_loader,
            args.test_epsilon,
            args.test_step_size,
            args.test_num_steps,
            True,
        )
        epoch_time = time.time() - start_time

        logger.info(
            "%d\t %d \t %.4f \t %.4f", epoch, epoch_time, nature_acc, robust_acc
        )

        writer.add_scalar("Test/Acc", float(nature_acc), epoch)
        writer.add_scalar("Test/Robust Acc", float(robust_acc), epoch)

        # Save checkpoint
        is_best = nature_acc > best_nature_acc
        best_nature_acc = max(nature_acc, best_nature_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "nature_acc": float(nature_acc),
                "robust_acc": float(robust_acc),
            },
            epoch,
            is_best,
            "nature",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

        is_best_robust = robust_acc > best_robust_acc
        best_robust_acc = max(robust_acc, best_robust_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "nature_acc": float(nature_acc),
                "robust_acc": float(robust_acc),
            },
            epoch,
            is_best_robust,
            "robust",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

    logger.info("Best Nature ACC %.4f", best_nature_acc)
    logger.info("Best Robust ACC %.4f", best_robust_acc)
    writer.close()


if __name__ == "__main__":
    main()