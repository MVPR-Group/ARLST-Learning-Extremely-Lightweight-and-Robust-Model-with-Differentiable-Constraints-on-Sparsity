import torch
import torch.nn as nn
import sys
import numpy as np
import logging
import math
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)
def get_lr_policy(lr_schedule):
    """Implement a new schduler directly in this file. 
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "constant": constant_schedule,
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    return d[lr_schedule]


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd,)
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, )
    else:
        print(f"{args.optimizer} is not supported.")
        sys.exit(0)
    return optim


def new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def constant_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.warmup_lr

        new_lr(optimizer, lr)

    return set_lr


def cosine_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs
            a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

        new_lr(optimizer, a)

    return set_lr


def step_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs

        a = lr
        if epoch >= 0.75 * epochs:
            a = lr * 0.1
        if epoch >= 0.9 * epochs:
            a = lr * 0.01
        if epoch >= epochs:
            a = lr * 0.001

        new_lr(optimizer, a)

    return set_lr

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))