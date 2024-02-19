import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.flickr30k_dataset import Flickr30kTrain, Flickr30kRetrievalEval
from data.pretrain_dataset import PretrainTrain
from transform.randaugment import RandomAugment

# Create dataset
def create_dataset(dataset, config, min_scale=0.5):
    # Transformation for train and evaluate
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # Load datasets
    if dataset == 'pretrain':
        train_dataset = PretrainTrain(transform_train, config['json_root'])
        return train_dataset
    elif dataset == 'retrieval_flickr':
        train_dataset = Flickr30kTrain(transform_train, config['image_root'], config['ann_root'])
        val_dataset = Flickr30kRetrievalEval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = Flickr30kRetrievalEval(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset

# Create DDP sampler
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers

# Create dataloaders
def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
