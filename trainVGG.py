from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import argparse
import importlib
import time
import logging
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import time
import models
import data
from argsVGG import parse_argsRLST
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
)
from utils.model import (
    get_layers,
    prepare_model_ARLST,
    initialize_scaled_score,
    scale_rand_init,
    show_gradients,
    current_model_pruned_fraction_ARLST,
    snip_init,
)


# TODO: update wrn, resnet mmodels. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning


def main():
    args = parse_argsRLST()
    parse_configs_file(args)

    # sanity checks
    if args.exp_mode in ["prune", "finetune"]:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.3f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Create model
    conv_layer, linear_layer = get_layers(args.layer_type)
    model = models.vgg.vgg16_bn_ARLST(conv_layer, linear_layer, args.init_type, num_classes=args.num_classes)
    model = model.to(device)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters:", num_params / 1000000, "M")
    logger.info(model)

    # prepare models for training/pruning/fine-tuning
    prepare_model_ARLST(model, args)

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # Dataloader
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()
    logger.info(args.dataset, D, len(train_loader.dataset), len(test_loader.dataset))

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    val = getattr(importlib.import_module("utils.eval"), args.val_method)

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))

    # Init scores once source net is loaded.
    # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
    if args.scaled_score_init:
        initialize_scaled_score(model)

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    # Whether implement snip init
    if args.snip_init:
        snip_init(model, criterion, optimizer, train_loader, device, args)

    # assert not args.source_net, (
    #     "Incorrect setup: "
    #     # "resume => required to resume a previous experiment (loads all parameters)|| "
    #     "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    # )

    # Evaluate
    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        # p1 = val(model, device, test_loader, criterion, args, writer)[0]
        prec1, _, _ = val(model, device, test_loader, criterion, args, writer)
        logger.info(f"Validation accuracy {args.val_method} for source-net: {prec1}")
        if args.evaluate:
            return

    best_prec1 = 0

    show_gradients(model)

    # Start training
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        a = time.time()
        lr_policy(epoch)  # adjust learning rate

        # train
        trainer(model, device, train_loader, criterion, optimizer, epoch, args, writer)

        # evaluate on test set
        if args.val_method == "base":
            prec1, _ = val(model, device, test_loader, criterion, args, writer, epoch)
        elif args.val_method == "adv":
            prec1, prec5, prec1_natural = val(model, device, test_loader, criterion, args, writer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )

        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1}, best_prec {best_prec1} ,prec1_natural{prec1_natural} "
        )

        if args.exp_mode in ["prune"]:
            logger.info(
                "Pruned model: {:.2f}%".format(
                    current_model_pruned_fraction_ARLST(
                        model,
                        os.path.join(result_sub_dir, "checkpoint"),
                        verbose=False,
                    )
                )
            )
        # clone results to latest subdir (sync after every epoch)
        # Latest_subdir: stores results from latest run of an experiment.
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        b = time.time()
        print("Time:", b - a)
    current_model_pruned_fraction_ARLST(
        model, os.path.join(result_sub_dir, "checkpoint"), verbose=True
    )


if __name__ == "__main__":
    main()
