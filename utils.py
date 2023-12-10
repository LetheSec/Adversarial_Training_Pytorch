import logging
import os
import random

import numpy as np
import torch


# 固定随即种子
def random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, logfile="output.log"):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="[%(asctime)s] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            filename=self.logfile,
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(
    state,
    epoch,
    is_best,
    which_best,
    save_path,
    save_freq=10,
):
    filename = os.path.join(save_path, "checkpoint_" + str(epoch) + ".tar")
    if epoch % save_freq == 0:
        if not os.path.exists(filename):
            torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(
            save_path, "best_" + str(which_best) + "_checkpoint.tar"
        )
        torch.save(state, best_filename)


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened**2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


import logging
import os
import random

import numpy as np
import torch


# 固定随即种子
def random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, logfile="output.log"):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="[%(asctime)s] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            filename=self.logfile,
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(
    state,
    epoch,
    is_best,
    which_best,
    save_path,
    save_freq=10,
):
    filename = os.path.join(save_path, "checkpoint_" + str(epoch) + ".tar")
    if epoch % save_freq == 0:
        if not os.path.exists(filename):
            torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(
            save_path, "best_" + str(which_best) + "_checkpoint.tar"
        )
        torch.save(state, best_filename)


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened**2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()