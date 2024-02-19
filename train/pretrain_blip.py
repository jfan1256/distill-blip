import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_pretrain import blip_pretrain
from train.utils import *
from data import create_dataset, create_sampler, create_loader

# Train model (per epoch)
def train(model, data_loader, optimizer, epoch, device, config):
    # Set model to train
    model.train()

    # Create metric loggers
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Set DDP sampler to different seed each time for different shuffle
    data_loader.sampler.set_epoch(epoch)

    # Iterate through images
    for i, (image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Warmup scheduler
        if epoch == 0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        # Zero optimizer gradients
        optimizer.zero_grad()

        # Put image to correct device
        image = image.to(device, non_blocking=True)

        # Ramp up alpha in the first 2 epochs
        alpha = config['alpha'] * min(1, (epoch * len(data_loader) + i) / (2 * len(data_loader)))

        # Train model
        loss_ita, loss_itm, loss_lm = model(image, caption, alpha=alpha)
        loss = loss_ita + loss_itm + loss_lm
        loss.backward()
        optimizer.step()

        # Update metric loggers
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

# Main function
def main(args, config):
    # Initialize distributed processes
    init_distributed_mode(args)

    # Get device
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Create dataset
    print("Creating dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],
                                                          samplers,
                                                          batch_size=[config['batch_size_train']] + [config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    # Create model
    print("Creating model")
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    # Load checkpoint
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('resume checkpoint from %s' % args.checkpoint)

    # Get model without DDP
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Start training
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['max_epoch']):
        # Step learning rate scheduler
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])

        # Train model
        train_stats = train(model, train_loader, optimizer, epoch, device, config)

        # Save model
        if is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Synchronize GPU processes
        dist.barrier()

    # Track time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # Arg Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/blip_pretrain.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # Get config
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Make output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get config
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # Execute main
    main(args, config)