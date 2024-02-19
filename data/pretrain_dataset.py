import json

from torch.utils.data import Dataset
from PIL import Image

from data.utils import pre_caption

class PretrainTrain(Dataset):
    def __init__(self, transform, json_root, max_words=30, prompt=''):
        with open(json_root, 'r') as file:
            self.pretrain_json = json.load(file)

        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.pretrain_json)

    def __getitem__(self, index):
        item = self.pretrain_json[index]

        image = Image.open(item['image_root']).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(item['caption'], self.max_words)

        return image, caption, item['id']