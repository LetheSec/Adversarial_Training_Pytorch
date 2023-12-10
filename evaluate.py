import argparse
import os
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchattacks
from torch.autograd import Variable
from torchvision import transforms, datasets
from utils import Logger

parser = argparse.ArgumentParser(description="Robust Evaluation")
parser.add_argument(
    "--arch",
    type=str,
    default="ResNet18",
    choices=[
        "WideResNet",
        "ResNet18",
    ],
)
# data
parser.add_argument(
    "--data", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"]
)
parser.add_argument(
    "--data-path", type=str, default="~/datasets/", help="where is the dataset CIFAR-10"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="checkpoint of pretrained model",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--result-dir",
    default="robust_evaluate",
    help="directory of model for saving checkpoint",
)
# data

args = parser.parse_args()

if args.data == "CIFAR100":
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

save_path = os.path.join(
    args.result_dir,
    args.data,
    args.arch,
)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
logger = Logger(os.path.join(save_path, "output.log"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_rank2_label(logit, y):
    batch_size = len(logit)
    tmp = logit.clone()
    tmp[torch.arange(batch_size), y] = -float("inf")
    return tmp.argmax(1)


def _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, mode):
    batch_size = len(X)
    with torch.no_grad():
        out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    if mode != "FGSM":
        random_noise = X.new(X.size()).uniform_(-epsilon, epsilon)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        adv_logit = model(X_pgd)
        if mode == "CW":
            rank2_label = get_rank2_label(adv_logit, y)
            loss = (
                -adv_logit[torch.arange(batch_size), y]
                + adv_logit[torch.arange(batch_size), rank2_label]
            )
            loss = loss.sum() / batch_size
        elif mode in ["PGD", "FGSM"]:
            loss = F.cross_entropy(adv_logit, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def clean_test(model, test_loader, device):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return float(test_accuracy)


def robust_test(model, test_loader, attack, device):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.enable_grad():
                adv_data = attack(data, target)
            output = model(adv_data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)

    return float(test_accuracy)


def eval_adv_test_whitebox(
    model, device, test_loader, epsilon, step_size, num_steps, mode
):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    num_steps = int(num_steps)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(
            model, X, y, epsilon, num_steps, step_size, mode
        )
        robust_err_total += err_robust
        natural_err_total += err_natural

    robust_acc = (len(test_loader.dataset) - robust_err_total) / len(
        test_loader.dataset
    )
    return float(robust_acc)


def robust_eval(model, test_loader, device):
    clean_acc = clean_test(model, test_loader, device)
    fgsm_acc = eval_adv_test_whitebox(
        model,
        device,
        test_loader,
        epsilon=8.0 / 255,
        step_size=8.0 / 255,
        num_steps=1,
        mode="FGSM",
    )
    pgd20_acc = eval_adv_test_whitebox(
        model,
        device,
        test_loader,
        epsilon=8.0 / 255,
        step_size=2.0 / 255,
        num_steps=20,
        mode="PGD",
    )
    pgd100_acc = eval_adv_test_whitebox(
        model,
        device,
        test_loader,
        epsilon=8.0 / 255,
        step_size=2.0 / 255,
        num_steps=100,
        mode="PGD",
    )
    cw100_acc = eval_adv_test_whitebox(
        model,
        device,
        test_loader,
        epsilon=8.0 / 255,
        step_size=2.0 / 255,
        num_steps=100,
        mode="CW",
    )

    # auto_attack = torchattacks.AutoAttack(
    #     model,
    #     norm="Linf",
    #     eps=8 / 255,
    #     version="standard",
    #     n_classes=10,
    #     seed=0,
    #     verbose=False,
    # )
    # aa_acc = robust_test(model, test_loader, auto_attack, device)

    results = {
        "Clean Acc": round(clean_acc, 4) * 100,
        "FGSM": round(fgsm_acc, 4) * 100,
        "PGD-20": round(pgd20_acc, 4) * 100,
        "PGD-100": round(pgd100_acc, 4) * 100,
        "CW-100": round(cw100_acc, 4) * 100,
        # "AutoAttack": round(aa_acc, 4) * 100,
    }
    return results


def main():
    # load model
    model = getattr(models, args.arch)(num_classes=NUM_CLASSES)
    # dataparallel
    model = nn.DataParallel(model).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    testset = getattr(datasets, args.data)(
        root=args.data_path, train=False, download=True, transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
    )
    eval_results = robust_eval(model, test_loader, device)

    logger.info(args.checkpoint)
    logger.info(eval_results)


if __name__ == "__main__":
    main()