import time

import torch
import torch.nn as nn
import torchvision

from utils.logging import AverageMeter, ProgressMeter
from utils.eval import accuracy
from utils.adv import trades_loss


# TODO: add adversarial accuracy.
def train(model, device, train_loader, criterion, optimizer, epoch, args, writer):
    print(
        " ->->->->->->->->->-> One epoch with Adversarial training (TRADES) <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    dataloader = train_loader

    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training data
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(f"Training images range: {[torch.min(images), torch.max(images)]}")

        output, Loss_L1, Loss_L2, Loss_L3 = model(images)

        # loss_reg = 0.0001*Loss_L1 + 1*Loss_L2
        # Only used when compressing more than 10x
        loss_reg = 0.0001 * Loss_L1 + 1 * Loss_L2 + 0.000001 * Loss_L3

        loss = loss_reg + trades_loss(
            model=model,
            x_natural=images,
            y=target,
            device=device,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            args=args,
            distance=args.distance,
        )

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write_to_tensorboard(
                writer, "train", epoch * len(train_loader) + i
            )

