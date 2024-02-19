import argparse
import numpy as np
import yaml
import random
import json
import torch.backends.cudnn as cudnn

from pathlib import Path

from models.blip_pretrain import blip_pretrain
from models.blip_retrieval import blip_retrieval
from models.dlip_caption import dlip_caption
from models.dlip_pretrain import DLIPPretrain, dlip_pretrain
from train.utils import *
from data import create_dataset, create_sampler, create_loader

# Train model (per epoch)
def train(model, data_loader, optimizer, epoch, device):
    # Set model to train
    model.train()

    # Metric Loggers
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_lm', SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # Print frequency
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10

    # # Set data_loader to a new random sample
    # data_loader.sampler.set_epoch(epoch)

    # Iterate through images
    for i, (image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Zero out gradients
        optimizer.zero_grad()

        # Transfer image to correct device
        image = image.to(device, non_blocking=True)

        # Train model and calculate loss
        loss_lm = model(image, caption)
        loss = loss_lm

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update metric loggers
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # # Train on partial data
        # if i == 50:
        #     break

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

# Main function to run
def main(args, config):
    # Initialize Multi-GPU Distributed Process
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
    datasets = [create_dataset(config['dataloader'], config, min_scale=0.2)][0]
    print('number of training samples: %d' % len(datasets[0]))

    # Get world size, rank, and create sampler (multi-gpu)
    num_tasks = get_world_size()
    global_rank = get_rank()
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)

    # Create dataloader
    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    print('Data Loader Size', len(data_loader))

    # Create model
    print("Creating model")
    model = dlip_caption(pretrained=config['pretrained'], med_config=config['med_config'], bert=config['bert'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    # Move model to correct device
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    # Load checkpoint
    start_epoch = 0

    # Store model without DDP for saving
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Start training
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['max_epoch']):
        # Step the learning rate
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        # Train model
        train_stats = train(model, data_loader, optimizer, epoch, device)

        # Save model and log results
        if is_main_process():
            log_stats = {
                          **{f'train_{k}': v for k, v in train_stats.items()},
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

        # Ensure all GPUs are synchronized
        if args.distributed:
            dist.barrier()

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # Parse configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/dlip_caption_flickr.yaml')
    parser.add_argument('--output_dir', default='output/DLIP_Caption_flickr')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # Get configs
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get configs
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # Execute main
    main(args, config)
