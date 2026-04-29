import os
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model import GRTSNet
from engine import train_one_epoch, evaluate
from dataset import get_dataset
from augmentation import get_transforms
from utils import setup_logger, save_checkpoint, load_checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from timm.scheduler import create_scheduler
import warnings

# ------------------------
# Ignore FutureWarnings
# ------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="timm")


# ------------------------
# Argument parser
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='GRTS-Net Training and Evaluation')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['uc_merced', 'aid'])
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--sched', type=str, default='cosine')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--drop-path', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--smoothing', type=float, default=0.02)
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--colorjitter', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local-rank', type=int, default=0)
    return parser.parse_args()


# ------------------------
# Main training function
# ------------------------
def main(rank, world_size, args):
    # ------------------------
    # Setup distributed
    # ------------------------
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    args.local_rank = rank
    args.world_size = world_size
    device = torch.device(f"cuda:{rank}")

    # Logger
    logger = setup_logger(args.output_dir, rank)
    if rank == 0:
        logger.info(f"World Size: {world_size}")

    # ------------------------
    # Dataset and DataLoader
    # ------------------------
    train_transform = get_transforms(args, train=True)
    val_transform = get_transforms(args, train=False)
    train_dataset, val_dataset, num_classes = get_dataset(args.data_dir, args.dataset, train_transform, val_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    # ------------------------
    # Model, Optimizer, Scheduler
    # ------------------------
    model = GRTSNet(num_classes=num_classes, feature_dim=768, dropout=args.dropout, drop_path=args.drop_path)
    print(model)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler(args, optimizer)

    start_epoch = 0
    best_top1 = 0.0
    

    print("Num_Classes = ", num_classes)
    print("Train data samples ", len(train_sampler))
    print("Validation data samples ", len(val_dataset))
    
    # Resume checkpoint if provided
    
    if args.resume:
        start_epoch, best_top1 = load_checkpoint(model, optimizer, scheduler, args.resume, logger)

    # ------------------------
    # Eval only
    # ------------------------
    if args.eval:
        top1, top5 = evaluate(model, val_loader, start_epoch, args, logger, device)
        if rank == 0:
            logger.info(f"Evaluation Top-1: {top1:.4f}  Top-5: {top5:.4f}")
        return

    # ------------------------
    # Training loop
    # ------------------------
    print("----- Start Training -----")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, scheduler, epoch, args, logger, device)

        # Evaluation after each epoch
        top1, top5 = evaluate(model, val_loader, epoch, args, logger, device)

        if rank == 0:
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint(model, optimizer, scheduler, epoch, top1, args.output_dir, is_best)
            logger.info(f"Epoch {epoch}: Top-1: {top1:.4f}  Top-5: {top5:.4f} | Best Top-1: {best_top1:.4f}")

    dist.destroy_process_group()


# ------------------------
# Entry point
# ------------------------
if __name__ == '__main__':
    args = parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
