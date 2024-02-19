import argparse
import pandas as pd
import yaml

from pathlib import Path
from tqdm import tqdm

from models.dlip_pretrain import dlip_pretrain
from train.utils import *
from data import create_dataset, create_sampler, create_loader

# Eval model
def eval(model, data_loader, device, batch_size):
    # Set model to train
    model.eval()

    # Store captions
    all_caption = []
    total = 0
    # Processing images in the dataloader
    start_time = time.time()
    for i, (image, caption, idx) in tqdm(enumerate(data_loader), desc="Evaluation"):
        # Generate DLIP caption
        image = image.to(device)
        dlip_caption = model.generate(image, sample=True)

        for id, dlip_cap, org_cap in zip(idx, dlip_caption, caption):
            all_caption.append([id.item(), dlip_cap, org_cap])

        if i == 4:
            total = 5 * batch_size
            break

    # Calculate and display speed
    elasped_time = time.time() - start_time
    image_per_ms = (elasped_time / total) * 1000
    print(f"Total time to produce caption for {total} images: {elasped_time} seconds")
    print(f"Average time to produce caption per image: {image_per_ms:.2f} milliseconds")

    evaluate = pd.DataFrame(all_caption, columns=['id', 'dlip_caption', 'original_caption'])
    evaluate = evaluate.set_index('id').sort_index()

    return evaluate

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
    model = dlip_pretrain(embed_dim=config['embed_dim'], med_config=config['med_config'], bert=config['bert'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

    # Move model to correct device
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    print('resume checkpoint from %s' % args.checkpoint)

    # Start evaluation
    print("Start eval")
    start_time = time.time()

    evaluate = eval(model, data_loader, device, config['batch_size'])
    for i in range(len(evaluate)):
        dlip_caption = evaluate.iloc[i]['dlip_caption']
        original_caption = evaluate.iloc[i]['original_caption']
        print(f"Row {i + 1:<3}: DLIP Caption: \"{dlip_caption:<80}\", Original Caption: \"{original_caption}\"")

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # Parse configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/dlip_pretrain_eval.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain_dlip_eval')
    parser.add_argument('--checkpoint', default='../save/dlip_pretrain_20.pth')
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
