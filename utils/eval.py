import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
from utils.logging import AverageMeter, ProgressMeter
from utils.adv import pgd_whitebox, fgsm, image_add_gaussian_noise, carlini_wagner_l2

# from autoattack.autoattack import AutoAttack
# from autoattack.square import SquareAttack
from scipy.stats import norm
import numpy as np
import time



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_1(output, target, device):
    with torch.no_grad():
        count = 0
        output = output.tolist()
        n_zero, n_one, n_two, n_three, n_four, n_five, n_six, n_seven, n_eight, n_nine = [], [], [], [], [], [], [], [], [], []
        t_zero, t_one, t_two, t_three, t_four, t_five, t_six, t_seven, t_eight, t_nine = [], [], [], [], [], [], [], [], [], []
        for index, j in enumerate(target):
            if j.item() == 0:
                t_zero.append(j)
                n_zero.append(output[count])
                count += 1
            elif j.item() == 1:
                t_one.append(j)
                n_one.append(output[count])
                count += 1
            elif j.item() == 2:
                t_two.append(j)
                n_two.append(output[count])
                count += 1
            elif j.item() == 3:
                t_three.append(j)
                n_three.append(output[count])
                count += 1
            elif j.item() == 4:
                t_four.append(j)
                n_four.append(output[count])
                count += 1
            elif j.item() == 5:
                t_five.append(j)
                n_five.append(output[count])
                count += 1
            elif j.item() == 6:
                t_six.append(j)
                n_six.append(output[count])
                count += 1
            elif j.item() == 7:
                t_seven.append(j)
                n_seven.append(output[count])
                count += 1
            elif j.item() == 8:
                t_eight.append(j)
                n_eight.append(output[count])
                count += 1
            elif j.item() == 9:
                t_nine.append(j)
                n_nine.append(output[count])
                count += 1

        n_zero = torch.tensor(n_zero).to(device)
        t_zero = torch.tensor(t_zero).to(device)
        n_one = torch.tensor(n_one).to(device)
        t_one = torch.tensor(t_one).to(device)
        n_two = torch.tensor(n_two).to(device)
        t_two = torch.tensor(t_two).to(device)
        n_three = torch.tensor(n_three).to(device)
        t_three = torch.tensor(t_three).to(device)
        n_four = torch.tensor(n_four).to(device)
        t_four = torch.tensor(t_four).to(device)
        n_five = torch.tensor(n_five).to(device)
        t_five = torch.tensor(t_five).to(device)
        n_six = torch.tensor(n_six).to(device)
        t_six = torch.tensor(t_six).to(device)
        n_seven = torch.tensor(n_seven).to(device)
        t_seven = torch.tensor(t_seven).to(device)
        n_eight = torch.tensor(n_eight).to(device)
        t_eight = torch.tensor(t_eight).to(device)
        n_nine = torch.tensor(n_nine).to(device)
        t_nine = torch.tensor(t_nine).to(device)

        _, pred0 = n_zero.topk(1, 1, True, True)
        pred0 = pred0.t()
        correct0 = pred0.eq(t_zero.view(1, -1).expand_as(pred0))

        _, pred1 = n_one.topk(1, 1, True, True)
        pred1 = pred1.t()
        correct1 = pred1.eq(t_one.view(1, -1).expand_as(pred1))

        _, pred2 = n_two.topk(1, 1, True, True)
        pred2 = pred2.t()
        correct2 = pred2.eq(t_two.view(1, -1).expand_as(pred2))

        _, pred3 = n_three.topk(1, 1, True, True)
        pred3 = pred3.t()
        correct3 = pred3.eq(t_three.view(1, -1).expand_as(pred3))

        _, pred4 = n_four.topk(1, 1, True, True)
        pred4 = pred4.t()
        correct4 = pred4.eq(t_four.view(1, -1).expand_as(pred4))

        _, pred5 = n_five.topk(1, 1, True, True)
        pred5 = pred5.t()
        correct5 = pred5.eq(t_five.view(1, -1).expand_as(pred5))

        _, pred6 = n_six.topk(1, 1, True, True)
        pred6 = pred6.t()
        correct6 = pred6.eq(t_six.view(1, -1).expand_as(pred6))

        _, pred7 = n_seven.topk(1, 1, True, True)
        pred7 = pred7.t()
        correct7 = pred7.eq(t_seven.view(1, -1).expand_as(pred7))

        _, pred8 = n_eight.topk(1, 1, True, True)
        pred8 = pred8.t()
        correct8 = pred8.eq(t_eight.view(1, -1).expand_as(pred8))

        _, pred9 = n_nine.topk(1, 1, True, True)
        pred9 = pred9.t()
        correct9 = pred9.eq(t_nine.view(1, -1).expand_as(pred9))

        # _ , pred = output.topk(1, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9 = [], [], [], [], [], [], [], [], [], []
        correct_k0 = correct0[:1].view(-1).float().sum(0, keepdim=True)
        correct_k1 = correct1[:1].view(-1).float().sum(0, keepdim=True)
        correct_k2 = correct2[:1].view(-1).float().sum(0, keepdim=True)
        correct_k3 = correct3[:1].view(-1).float().sum(0, keepdim=True)
        correct_k4 = correct4[:1].view(-1).float().sum(0, keepdim=True)
        correct_k5 = correct5[:1].view(-1).float().sum(0, keepdim=True)
        correct_k6 = correct6[:1].view(-1).float().sum(0, keepdim=True)
        correct_k7 = correct7[:1].view(-1).float().sum(0, keepdim=True)
        correct_k8 = correct8[:1].view(-1).float().sum(0, keepdim=True)
        correct_k9 = correct9[:1].view(-1).float().sum(0, keepdim=True)
        acc_0.append(correct_k0.mul_(100.0 / n_zero.size(0)))
        acc_1.append(correct_k1.mul_(100.0 / n_one.size(0)))
        acc_2.append(correct_k2.mul_(100.0 / n_two.size(0)))
        acc_3.append(correct_k3.mul_(100.0 / n_three.size(0)))
        acc_4.append(correct_k4.mul_(100.0 / n_four.size(0)))
        acc_5.append(correct_k5.mul_(100.0 / n_five.size(0)))
        acc_6.append(correct_k6.mul_(100.0 / n_six.size(0)))
        acc_7.append(correct_k7.mul_(100.0 / n_seven.size(0)))
        acc_8.append(correct_k8.mul_(100.0 / n_eight.size(0)))
        acc_9.append(correct_k9.mul_(100.0 / n_nine.size(0)))

    return acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9


def base(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)[0]
            output = output.squeeze()
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "test-images",
                    torchvision.utils.make_grid(images[0: len(images) // 4]),
                )
        progress.display(i)  # print final results

    return top1.avg, top5.avg


def adv(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluate on adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    adv_top1 = AverageMeter("Adv-pgd-Acc_1", ":6.2f")
    adv_top5 = AverageMeter("Adv-pgd-Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [losses, adv_losses, top1, top5, adv_top1, adv_top5],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)[0]
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # adversarial images
            images = pgd_whitebox(
                model,
                images,
                target,
                device,
                args.epsilon,
                args.num_steps,
                args.step_size,
                args.clip_min,
                args.clip_max,
                is_random=not args.const_init,
            )

            # compute output
            output = model(images)[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

        progress.display(i)  # print final results

    return adv_top1.avg, adv_top5.avg, top1.avg
