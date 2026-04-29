import torch
import torch.nn.functional as F
import torch.distributed as dist
from timm.utils import accuracy
from model import grts_loss
from mixup import mixup_data, cutmix_data
import time

# ------------------------
# AverageMeter for metrics
# ------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


# ------------------------
# Training for 1 epoch with ETA
# ------------------------
def train_one_epoch(model, loader, optimizer, scheduler, epoch, args, logger, device):
    model.train()
    loss_meter = AverageMeter()
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ------------------------
        # Mixup / CutMix
        # ------------------------
        if args.mixup > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup)
            def mix_loss_fn(loss_fn):
                return lam * loss_fn(labels_a) + (1 - lam) * loss_fn(labels_b)

        elif args.cutmix > 0:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, args.cutmix)
            def mix_loss_fn(loss_fn):
                return lam * loss_fn(labels_a) + (1 - lam) * loss_fn(labels_b)

        else:
            def mix_loss_fn(loss_fn):
                return loss_fn(labels)

        # ------------------------
        # Forward + Loss
        # ------------------------
        optimizer.zero_grad(set_to_none=True)
        final_out, aux_out, s_feats, t_feats = model(images)

        def base_loss(labs):
            return grts_loss(final_out, aux_out, s_feats, t_feats, labs, args)

        loss = mix_loss_fn(base_loss)
        loss.backward()

        # Gradient clipping
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

        # ------------------------
        # Track metrics
        # ------------------------
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)

        # Estimate ETA every 50 batches
        if batch_idx % 50 == 0 and args.local_rank == 0:
            elapsed = time.time() - start_time
            batches_done = batch_idx + 1
            batches_left = len(loader) - batches_done
            eta_sec = (elapsed / batches_done) * batches_left
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"Loss: {loss_meter.val:.4f}  Avg Loss: {loss_meter.avg:.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}  ETA: {eta_str}"
            )

    # ------------------------
    # Sync loss across GPUs
    # ------------------------
    total_loss = torch.tensor([loss_meter.sum], device=device)
    total_count = torch.tensor([loss_meter.count], device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / total_count).item()

    if args.local_rank == 0:
        logger.info(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

    # Step scheduler per epoch
    if scheduler is not None:
        scheduler.step(epoch)

    return avg_loss


# ------------------------
# Evaluation (Top-1 & Top-5)
# ------------------------
@torch.no_grad()
def evaluate(model, loader, epoch, args, logger, device):
    model.eval()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        final_out, _, _, _ = model(images)
        acc1, acc5 = accuracy(final_out, labels, topk=(1, 5))

        bs = images.size(0)
        top1_meter.update(acc1.item(), bs)
        top5_meter.update(acc5.item(), bs)

    # ------------------------
    # Sync across GPUs
    # ------------------------
    top1_sum = torch.tensor([top1_meter.sum], device=device)
    top5_sum = torch.tensor([top5_meter.sum], device=device)
    total_count = torch.tensor([top1_meter.count], device=device)

    dist.all_reduce(top1_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(top5_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    top1 = (top1_sum / total_count).item()
    top5 = (top5_sum / total_count).item()

    if args.local_rank == 0:
        logger.info(f"Epoch {epoch} Validation: Top-1: {top1:.4f}  Top-5: {top5:.4f}")

    return top1, top5
