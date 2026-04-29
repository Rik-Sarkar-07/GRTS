import os
import logging
import torch

def setup_logger(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('GRTS-Net')
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    handler = logging.FileHandler(os.path.join(output_dir, 'train_log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
    return logger

def save_checkpoint(model, optimizer, scheduler, epoch, acc, output_dir, is_best=False):
    state = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch + 1,  # Next epoch
        'best_acc': acc,
    }
    torch.save(state, os.path.join(output_dir, 'checkpoint_current.pth'))
    if is_best:
        torch.save(state, os.path.join(output_dir, 'checkpoint_best.pth'))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, logger):
    state = torch.load(checkpoint_path, map_location='cpu')
    model.module.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    start_epoch = state['epoch']
    best_acc = state.get('best_acc', 0.0)
    logger.info(f"Resumed from {checkpoint_path} at epoch {start_epoch - 1}, best_acc {best_acc}")
    return start_epoch, best_acc