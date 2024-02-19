import argparse
import pandas as pd
import yaml

from pathlib import Path
from tqdm import tqdm

from models.dlip_caption import dlip_caption
from models.dlip_retrieval import dlip_retrieval
from train.utils import *
from data import create_dataset, create_sampler, create_loader

# Eval model
def eval(model, data_loader, device, batch_size):
    # Set model to eval
    torch.backends.cudnn.benchmark = True
    model.eval()

    # Store captions
    all_caption = []
    total = 0
    start_time = 0
    elasped_time = 0
    image_per_ms = 0

    # Processing images in the dataloader
    for i, (image, caption, idx) in tqdm(enumerate(data_loader), desc="Evaluation"):
        if i == 0:
            start_time = time.time()
        # Generate DLIP caption
        image = image.to(device)
        image_feat = model.get_image_feat(image)
        text_feat = model.get_text_feat(image, caption)

        if i == 4:
            total = 5 * batch_size
            elasped_time = time.time() - start_time
            image_per_ms = (elasped_time / total) * 1000
            break

    # Calculate and display speed
    print(f"Total time to produce caption for {total} images: {elasped_time} seconds")
    print(f"Average time to produce caption per image: {image_per_ms:.2f} milliseconds")

# Main function to run
def main(args, config):
    # Get device
    device = torch.device(args.device)

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
    model = dlip_retrieval(pretrained=config['pretrained'], med_config=config['med_config'],
                           image_size=config['image_size'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                           queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                           bert=config['bert'])
    # Move model to correct device
    model = model.to(device)

    # Start evaluation
    print("Start eval")
    start_time = time.time()

    eval(model, data_loader, device, config['batch_size'])

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # Parse configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/dlip_retrieval_flickr_eval.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_dlip_eval')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cpu')
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
