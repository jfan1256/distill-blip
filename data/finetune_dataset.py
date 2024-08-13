import os
import json
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from data.utils import pre_caption

class FinetuneTrain(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=100, prompt=''):
        # Load the tags
        self.tag_flickr30k = pd.read_csv('../export/share/datasets/tag/id_and_tag_flickr30k.csv')
        self.tag_flickr30k = self.tag_flickr30k.set_index('id')
        self.tag_textcaps = pd.read_csv('../export/share/datasets/tag/id_and_tag_textcaps.csv')
        self.tag_textcaps = self.tag_textcaps.set_index('id')
        self.tag_coco = pd.read_csv('../export/share/datasets/tag/id_and_tag_coco.csv')
        self.tag_coco = self.tag_coco.set_index('id')
        self.tag = pd.concat([self.tag_flickr30k, self.tag_textcaps, self.tag_coco], axis=0)

        # Initialize required attributes
        filename = 'finetune_train.json'
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        # Maintain a dictionary to manage unique image IDs
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        # Initialize the count dictionary for tracking whether an image_id has been processed
        self.img_ids_count = {key: 0 for key in self.img_ids.keys()}

        # List to hold new annotations to be added after iterating
        new_annotations = []

        # Iterate over the annotations to create tag captions
        for ann in self.annotation:
            img_id = ann['image_id']
            img_filename = os.path.basename(ann['image'])

            # Check if the img_id has already been processed
            if self.img_ids_count[img_id] == 0:
                if img_filename in self.tag.index:
                    tags = self.tag.loc[self.tag.index == img_filename].values[0].tolist()
                    tags = [tag for tag in tags if isinstance(tag, str) and not pd.isna(tag)]
                    tag_caption = f"{', '.join(tags[:-1])}, and {tags[-1]}"

                    new_annotation = {
                        "image": ann['image'],
                        "caption": tag_caption,
                        "image_id": img_id
                    }
                    new_annotations.append(new_annotation)

                # Mark this img_id as processed
                self.img_ids_count[img_id] = 1

        # Extend the original annotation list with the new annotations
        self.annotation.extend(new_annotations)
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # This order is defined in configs
        # self.image_root[1] = TextCaps
        # self.image_root[0] = Flickr30k
        # self.image_root[2] = COCO
        if 'train_images' in ann['image']:
            image_path = os.path.join(self.image_root[1], ann['image'])
        elif 'flickr30k-images' in ann['image']:
            image_path = os.path.join(self.image_root[0], ann['image'])
        elif 'train2014' in ann['image']:
            image_path = os.path.join(self.image_root[2], ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]

class FinetuneRetrievalEval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=100):
        filenames = {'val': 'finetune_val.json', 'test': 'finetune_test.json'}

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # This order is defined in configs
        # self.image_root[1] = TextCaps
        # self.image_root[0] = Flickr30k
        # self.image_root[2] = COCO
        if 'train_images' in ann['image']:
            image_path = os.path.join(self.image_root[1], ann['image'])
        elif 'flickr30k-images' in ann['image']:
            image_path = os.path.join(self.image_root[0], ann['image'])
        elif 'val2014' in ann['image']:
            image_path = os.path.join(self.image_root[2], ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index