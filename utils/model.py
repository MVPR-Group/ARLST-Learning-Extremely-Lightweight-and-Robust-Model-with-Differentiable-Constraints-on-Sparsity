import torch
import torch.nn as nn
import torchvision

import os
import math
import numpy as np

from models.layers import SubnetConv, Linear_ARLST


# TODO: avoid freezing bn_params
# Some mutils are borrowed from https://github.com/allenai/hidden-networks
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores", "popup_scores_A", "popup_scores_B", "A", "B"]
    # assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores", "popup_scores_A", "popup_scores_B", "A", "B"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "arlst":
        return SubnetConv, Linear_ARLST
    else:
        raise ValueError("Incorrect layer type")


def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def snip_init(model, criterion, optimizer, train_loader, device, args):
    print("Using SNIP initialization")
    assert args.exp_mode == "pretrain"
    optimizer.zero_grad()
    # init the score with kaiming normal init
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            nn.init.kaiming_normal_(m.popup_scores, mode="fan_in")

    set_prune_rate_model(model, 1.0)
    unfreeze_vars(model, "popup_scores")

    # take a forward pass and get gradients
    for _, data in enumerate(train_loader):
        images, target = data[0].to(device), data[1].to(device)

        output = model(images)
        loss = criterion(output, target)

        loss.backward()
        break

    # update scores with their respective connection sensitivty
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            print(m.popup_scores.data)
            m.popup_scores.data = m.popup_scores.grad.data.abs()
            print(m.popup_scores.data)

    # update k back to args.k.
    set_prune_rate_model(model, args.k)
    freeze_vars(model, "popup_scores")


def initialize_scores(model, init_type):
    #     print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
        if hasattr(m, "popup_scores_A"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
        if hasattr(m, "popup_scores_B"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                    math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )

        if hasattr(m, "popup_scores_A"):
            n = nn.init._calculate_correct_fan(m.popup_scores_A, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores_A.data = (
                    math.sqrt(6 / n) * m.A.data / torch.max(torch.abs(m.A.data))
            )

        if hasattr(m, "popup_scores_B"):
            n = nn.init._calculate_correct_fan(m.popup_scores_B, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores_B.data = (
                    math.sqrt(6 / n) * m.B.data / torch.max(torch.abs(m.B.data))
            )


def scale_rand_init(model, k):
    print(
        f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # print(f"previous std = {torch.std(m.weight.data)}")
            m.weight.data = 1 / math.sqrt(k) * m.weight.data
            # print(f"new std = {torch.std(m.weight.data)}")


def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """

    set_prune_rate_model(model, args.k)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None  | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        freeze_vars(model, "popup_scores_A", args.freeze_bn)
        freeze_vars(model, "popup_scores_B", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        unfreeze_vars(model, "A_ref")
        unfreeze_vars(model, "B_ref")

    elif args.exp_mode == "prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")

        unfreeze_vars(model, "popup_scores")
        unfreeze_vars(model, "popup_scores_A")
        unfreeze_vars(model, "popup_scores_B")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)
        freeze_vars(model, "A_ref", args.freeze_bn)
        freeze_vars(model, "B_ref", args.freeze_bn)

    elif args.exp_mode == "finetune":
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        freeze_vars(model, "popup_scores_A", args.freeze_bn)
        freeze_vars(model, "popup_scores_B", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        unfreeze_vars(model, "A_ref")
        unfreeze_vars(model, "B_ref")

    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scores(model, args.scores_init_type)


def prepare_model_ARLST(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """

    set_prune_rate_model(model, args.k)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None  | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        freeze_vars(model, "popup_scores_A", args.freeze_bn)
        freeze_vars(model, "popup_scores_B", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        unfreeze_vars(model, "A")
        unfreeze_vars(model, "B")

    elif args.exp_mode == "prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")

        unfreeze_vars(model, "popup_scores")
        unfreeze_vars(model, "popup_scores_A")
        unfreeze_vars(model, "popup_scores_B")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)
        freeze_vars(model, "A", args.freeze_bn)
        freeze_vars(model, "B", args.freeze_bn)

    elif args.exp_mode == "finetune":
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        freeze_vars(model, "popup_scores_A", args.freeze_bn)
        freeze_vars(model, "popup_scores_B", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
        unfreeze_vars(model, "A")
        unfreeze_vars(model, "B")

    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scores(model, args.scores_init_type)


def current_model_pruned_fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense mmodels
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    if verbose:
                        print(i, v, p)

        return np.mean(pl)


def current_model_pruned_fraction_ARLST(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense models
    path = os.path.join(result_dir, "model_best.pth.tar")

    pl = []

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if i == "module.classifier3":
                continue
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)

            if isinstance(v, nn.Module):
                if i + ".A" in state_dict.keys():
                    d = state_dict[i + ".A"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)

                if i + ".B" in state_dict.keys():
                    d = state_dict[i + ".B"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)

        return np.mean(pl)





