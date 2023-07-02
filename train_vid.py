import datetime
import os
from statistics import mode
import sys
import time
import random
import setproctitle
from typing import Iterable
from shutil import copyfile

import torch
import torch.utils.data
from torchvision import transforms

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


from dataloader.video import YoutubeVOS
from models import resnet
from models.model import HVC
import core
from core.clip_sampler import RandomClipSamplerFrame
from core import transform_coord
from core.config import get_arguments


total_step = 0


def main(args):
    # set name
    setproctitle.setproctitle("self-supervised vos")
    
    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    # ============ preparing data ============
    # simple augmentation
    transform_1 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(args.img_size, scale=(args.min_crop, args.max_crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    transform_2 = transform_1
    transform_train = (transform_1, transform_2)

    dataset = YoutubeVOS(
        args.data_path,
        frames_per_clip=args.clip_len,
        frame_rate=args.frame_rate,
        transform=transform_train)

    train_sampler = RandomClipSamplerFrame(dataset.resampling_idxs, args.clips_per_video)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True)
    args.num_steps = len(data_loader)
    print("Number of video frames = %d" % len(dataset))
    print('Number of training steps per epoch = %d' % len(data_loader))

    if args.enable_wandb:
        wandb_logger = core.WandbLogger(args)
    else:
        wandb_logger = None

    # ============ building hybrid visual correspondence network ============
    encoder = resnet.__dict__[args.arch]
    model = HVC(encoder, args)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.enable_amp == True:
        scaler = GradScaler()
    else:
        scaler = None

    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, 'hvc.pth')
        if os.path.exists(resume_file):
            print(f'Auto resume from {resume_file}')
            args.resume = resume_file
        else:
            print(f'No checkpoint found in {args.output_dir}, igoring auto resume')

    if args.resume:
        assert os.path.isfile(args.resume)
        load_model(args, model, optimizer)
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs+1):
        train_one_epoch(model, data_loader, scaler,
                        optimizer, device, epoch, args,
                        wandb_logger=wandb_logger)
        if args.output_dir and (epoch == args.epochs or epoch % args.save_ckpt_freq == 0):
            save_model(args, epoch, model, optimizer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    scaler,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args=None, wandb_logger=None):

    global total_step
    model.train(True)
    metric_logger = core.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', core.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', core.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for step, (videos, coords) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_step += 1
        start_time = time.time()

        video1 = videos[0].to(device, non_blocking=True)
        video2 = videos[1].to(device, non_blocking=True)
        coord1 = coords[0].to(device, non_blocking=True)
        coord2 = coords[1].to(device, non_blocking=True)

        if scaler is not None:
            with autocast():
                loss = model(video1, video2, coord1, coord2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(video1, video2, coord1, coord2)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(video1.shape[0] / (time.time() - start_time))

        if wandb_logger is not None and total_step % 100 == 0:
            wandb_logger.log(dict(loss=loss.item()))
            wandb_logger.log(dict(learning_rate=optimizer.param_groups[0]['lr']))


def save_model(args, epoch, model, optimizer):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args
    }
    ckpt_name = os.path.join(args.output_dir, f'hvc_epoch_{epoch}.pth')
    torch.save(to_save, ckpt_name)
    copyfile(ckpt_name, os.path.join(args.output_dir, 'hvc.pth'))


def load_model(args, model, optimizer):
    print(f"Loading checkpoint {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Loaded successfully checkpoint {args.resume}")
    del checkpoint
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])  
    main(args)
