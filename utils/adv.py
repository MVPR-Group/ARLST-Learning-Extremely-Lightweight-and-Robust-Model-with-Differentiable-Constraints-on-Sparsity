import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import skimage
import numpy as np
import torch
import copy
# from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
# from autoattack.autoattack import AutoAttack
# from autoattack.square import SquareAttack


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


# def compute_jacobian(model, input):
#
#     output = model(input)
#
#     num_features = int(np.prod(input.shape[1:]))
#     jacobian = torch.zeros([output.size()[1], num_features])
#     mask = torch.zeros(output.size())  # chooses the derivative to be calculated
#     for i in range(output.size()[1]):
#         mask[:, i] = 1
#         zero_gradients(input)
#         output.backward(mask, retain_graph=True)
#         # copy the derivative to the target place
#         jacobian[i] = input._grad.squeeze().view(-1, num_features).clone()
#         mask[:, i] = 0  # reset
#
#     return jacobian


def saliency_map(jacobian, target_index, increasing, search_space, nb_features):
    domain = torch.eq(search_space, 1).float()  # The search domain
    # the sum of all features' derivative with respect to each class
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]  # The forward derivative of the target class
    others_grad = all_sum - target_grad  # The sum of forward derivative of other classes

    # this list blanks out those that are not in the search domain
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float()
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float()
    increase_coef = increase_coef.view(-1, nb_features)

    # calculate sum of target forward derivative of any 2 features.
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features,
                                                                  1)  # PyTorch will automatically extend the dimensions
    # calculate sum of other forward derivative of any 2 features.
    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    # zero out the situation where a feature sums with itself
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte()

    # According to the definition of saliency map in the paper (formulas 8 and 9),
    # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)
    # apply the mask to the saliency map
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    # do the multiplication according to formula 10 in the paper
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    # get the most significant two pixels
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q


# ref: https://github.com/yaodongyu/TRADES
def trades_loss(
        model,
        x_natural,
        y,
        device,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        args,
        distance="l_inf",
        natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach())
    if args.train_attack_methods == "square":
        adversary = SquareAttack(model, eps=0.031, device=device)
        x_adv = adversary.perturb(x_natural, y)
    elif args.train_attack_methods == "auto":
        adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)
        x_adv = adversary.run_standard_evaluation(x_natural, y, bs=batch_size)
    elif args.train_attack_methods == "cw":
        x_adv = carlini_wagner_l2(model, x_natural, args.num_classes, args, y)
    else:
        if distance == "l_inf":
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(
                        F.log_softmax(model(x_adv)[0], dim=1),
                        F.softmax(model(x_natural)[0], dim=1),
                    )
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(
                    torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
                )
                x_adv = torch.clamp(x_adv, clip_min, clip_max)
        elif distance == "l_2":
            delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(
                        F.log_softmax(model(adv)[0], dim=1), F.softmax(model(x_natural)[0], dim=1)
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
                delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)[0]

    loss_natural = natural_criterion(logits, y)

    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv)[0], dim=1),
                                                    F.softmax(model(x_natural)[0], dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def trades_loss_VIT(
        model,
        x_natural,
        y,
        device,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        distance="l_inf",
        natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
            x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
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
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv)[0], dim=1), F.softmax(model(x_natural)[0], dim=1)
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
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def trades_loss_hot_vector(
        model,
        x_natural,
        y,
        device,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        distance="l_inf",
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
            x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
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
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
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
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
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


# TODO: support L-2 attacks too.
def pgd_whitebox(
        model,
        x,
        y,
        device,
        epsilon,
        num_steps,
        step_size,
        clip_min,
        clip_max,
        is_random=True,
):
    x_pgd = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([x_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            # VIT
            # loss = nn.CrossEntropyLoss()(model(x_pgd)[0], y)

            loss = nn.CrossEntropyLoss()(model(x_pgd)[0], y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, clip_min, clip_max), requires_grad=True)

    return x_pgd


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def image_add_gaussian_noise(images):
    images = skimage.util.random_noise(images.cpu(), mode="gaussian", var=0.1)
    return images


def trades_loss_Gaussian(
        model,
        x_natural,
        y,
        device,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        distance="l_inf",
        natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    x_adv = torch.FloatTensor(image_add_gaussian_noise(x_natural)).to(device)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


import torch

INF = float("inf")


def carlini_wagner_l2(
        model_fn,
        x,
        n_classes,
        args,
        y=None,
        targeted=False,
        lr=5e-3,
        confidence=0,
        clip_min=0,
        clip_max=1,
        initial_const=1e-2,
        binary_search_steps=5,
        max_iterations=1000,
):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    :param model_fn: a callable that takes an input tensor and returns
              the model logits. The logits should be a tensor of shape
              (n_examples, n_classes).
    :param x: input tensor of shape (n_examples, ...), where ... can
              be any arbitrary dimension that is compatible with
              model_fn.
    :param n_classes: the number of classes.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when
              crafting adversarial samples. Otherwise, model predictions
              are used as labels to avoid the "label leaking" effect
              (explained in this paper:
              https://arxiv.org/abs/1611.01236). If provide y, it
              should be a 1D tensor of shape (n_examples, ).
              Default is None.
    :param targeted: (optional) bool. Is the attack targeted or
              untargeted? Untargeted, the default, will try to make the
              label incorrect. Targeted will instead try to move in the
              direction of being more like y.
    :param lr: (optional) float. The learning rate for the attack
              algorithm. Default is 5e-3.
    :param confidence: (optional) float. Confidence of adversarial
              examples: higher produces examples with larger l2
              distortion, but more strongly classified as adversarial.
              Default is 0.
    :param clip_min: (optional) float. Minimum float value for
              adversarial example components. Default is 0.
    :param clip_max: (optional) float. Maximum float value for
              adversarial example components. Default is 1.
    :param initial_const: The initial tradeoff-constant to use to tune the
              relative importance of size of the perturbation and
              confidence of classification. If binary_search_steps is
              large, the initial constant is not important. A smaller
              value of this constant gives lower distortion results.
              Default is 1e-2.
    :param binary_search_steps: (optional) int. The number of times we
              perform binary search to find the optimal tradeoff-constant
              between norm of the perturbation and confidence of the
              classification. Default is 5.
    :param max_iterations: (optional) int. The maximum number of
              iterations. Setting this to a larger value will produce
              lower distortion results. Using only a few iterations
              requires a larger learning rate, and will produce larger
              distortion results. Default is 1000.
    """

    def compare(pred, label, is_logits=False):
        """
        A helper function to compare prediction against a label.
        Returns true if the attack is considered successful.
        :param pred: can be either a 1D tensor of logits or a predicted
                class (int).
        :param label: int. A label to compare against.
        :param is_logits: (optional) bool. If True, treat pred as an
                array of logits. Default is False.
        """

        # Convert logits to predicted class if necessary
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)

        return pred == label if targeted else pred != label

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        pred = model_fn(x)
        y = torch.argmax(pred, 1)

    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    # Initialize some values needed for binary search on const
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const

    o_bestl2 = [INF] * len(x)
    o_bestscore = [-1.0] * len(x)
    x = torch.clamp(x, clip_min, clip_max)
    ox = x.clone().detach()  # save the original x
    o_bestattack = x.clone().detach()

    # Map images into the tanh-space
    x = (x - clip_min) / (clip_max - clip_min)
    x = torch.clamp(x, 0, 1)
    x = x * 2 - 1
    x = x.cpu().numpy()
    x = np.arctanh(x * 0.999999)
    x = torch.from_numpy(x).cuda(gpu_list[0])

    # Prepare some variables
    modifier = torch.zeros_like(x, requires_grad=True)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float)

    # Define loss functions and optimizer
    f_fn = lambda real, other, targeted: torch.max(
        ((other - real) if targeted else (real - other)) + confidence,
        torch.tensor(0.0).to(real.device),
    )
    l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
    optimizer = torch.optim.Adam([modifier], lr=lr)

    # Outer loop performing binary search on const
    for outer_step in range(binary_search_steps):
        # Initialize some values needed for the inner loop
        bestl2 = [INF] * len(x)
        bestscore = [-1.0] * len(x)

        # Inner loop performing attack iterations
        for i in range(max_iterations):
            # One attack step
            new_x = (torch.tanh(modifier + x) + 1) / 2
            new_x = new_x * (clip_max - clip_min) + clip_min
            logits = model_fn(new_x)

            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

            optimizer.zero_grad()
            f = f_fn(real, other, targeted)
            l2 = l2dist_fn(new_x, ox)
            loss = (const * f + l2).sum()
            loss.backward()
            optimizer.step()

            # Update best results
            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                y_n = y[n]
                succeeded = compare(logits_n, y_n, is_logits=True)
                if l2_n < o_bestl2[n] and succeeded:
                    pred_n = torch.argmax(logits_n)
                    o_bestl2[n] = l2_n
                    o_bestscore[n] = pred_n
                    o_bestattack[n] = new_x_n
                    # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                    bestl2[n] = l2_n
                    bestscore[n] = pred_n
                elif l2_n < bestl2[n] and succeeded:
                    bestl2[n] = l2_n
                    bestscore[n] = torch.argmax(logits_n)

        # Binary search step
        for n in range(len(x)):
            y_n = y[n]

            if compare(bestscore[n], y_n) and bestscore[n] != -1:
                # Success, divide const by two
                upper_bound[n] = min(upper_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
            else:
                # Failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lower_bound[n] = max(lower_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    const[n] *= 10

    return o_bestattack.detach()
