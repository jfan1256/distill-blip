import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from torch import nn
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import BertTokenizer

from models.vit import VisionTransformer
from models.med import BertConfig, BertModel, BertLMHeadModel

class DLIPProd(nn.Module):
    def __init__(self,
                 name: str = 'caption',
                 weight: str = '../save/dlip_retrieval_flickr.pth',
                 device: str = 'cpu',
                 ):
        
        '''
        name: Name of model to use (either 'caption' or 'retrieval'
        weight: Directory str path for model weights following the format {'model': <torch model state dict>}
        device: Str torch device to use for model computation (either 'cpu' or 'cuda')
        '''

        # Required assertion
        if name not in ['caption', 'retrieval']:
            raise ValueError('Name should be either "caption" or "retrieval."')
        if device not in ['cpu', 'cuda']:
            raise ValueError('Device should be either "cpu" or "cuda".')
        if not os.path.isfile(weight):
            raise FileNotFoundError("File {} does not exist.".format(weight))

        # Super 
        super().__init__()

        # Initialize device
        self.device = torch.device(device)

        # Initialize weight
        self.weight = weight

        # Initialize Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        # Initialize Visual Encoder
        vision_width = 384
        self.visual_encoder = VisionTransformer(img_size=384, patch_size=16, embed_dim=vision_width, depth=12, num_heads=6, use_grad_checkpointing=True, ckpt_layer=4, drop_path_rate=0)

        # Initialize Text Encoder or Text Decoder
        bert_config = {"architectures": ["BertModel"], "attention_probs_dropout_prob": 0.1, "hidden_act": "gelu", "hidden_dropout_prob": 0.1, "hidden_size": 384, "initializer_range": 0.02,
                       "intermediate_size": 3072, "layer_norm_eps": 1e-12, "max_position_embeddings": 512, "model_type": "bert", "num_attention_heads": 12, "num_hidden_layers": 6,
                       "pad_token_id": 0, "type_vocab_size": 2, "vocab_size": 30524, "encoder_width": 384, "add_cross_attention": True}
        bert_config = BertConfig(**bert_config)
        bert_config.encoder_width = vision_width
        
        if name == 'caption':
            self.text_decoder = BertLMHeadModel(config=bert_config)
        elif name == 'retrieval':
            self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        # Initialize Vision and Text Projection
        if name == 'retrieval':
            self.vision_proj = nn.Linear(vision_width, 256)
            self.text_proj = nn.Linear(vision_width, 256)
        
        # Initialize Trained Model
        self.load_model()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------LOAD MODEL------------------------------------------------------------------------------
    # Interpolate Position Embedding for Visual Encoder
    @staticmethod
    def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
        # interpolate position embedding
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = visual_encoder.patch_embed.num_patches
        num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            return new_pos_embed
        else:
            return pos_embed_checkpoint

    # Load Model
    def load_model(self):
        self.to(self.device)
        checkpoint = torch.load(self.weight, map_location=self.device)
        state_dict = checkpoint['model']
        state_dict['visual_encoder.pos_embed'] = self.interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], self.visual_encoder)
        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------HANDLE IMAGE---------------------------------------------------------------------------------
    # Handle image cases (PIL, CV2, or NumPy)
    @staticmethod
    def handle_case(transform, image):
        # CV2 image or NumPy array
        if isinstance(image, np.ndarray):
            # Convert from BGR (OpenCV) to RGB (if necessary)
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError("Unsupported image format. Please provide a PIL Image, CV2 image, or NumPy array.")
        return transform(image)

    # Handles images (image can be a list of images or a single image)
    def handle_image(self, image_input):
        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # If input is a list of images
        if isinstance(image_input, list):
            processed_images = [self.handle_case(transform, image) for image in image_input]
            # Stack all processed images into a single tensor
            image_tensor = torch.stack(processed_images)
        else:
            # Process a single image
            image_tensor = self.handle_case(transform, image_input)
            # Ensure the tensor is 4D [1, C, H, W]
            image_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
        return image_tensor

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------CORE------------------------------------------------------------------------------------
    # Generate caption (returns a list of captions [len(list) > 0])
    def generate(self, image, max_length=30, min_length=10, top_p=0.75):
        # Handle Image
        image = self.handle_image(image)

        # Get image embeddings
        image_embed = self.visual_encoder(image)
        image_attn = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(self.device)

        # Prompt input id
        prompt = ['a picture of '] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        # Nucleus sampling
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                             max_length=max_length,
                                             min_length=min_length,
                                             do_sample=True,
                                             top_p=top_p,
                                             num_return_sequences=1,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             repetition_penalty=1.1,
                                             encoder_hidden_states=image_embed,
                                             encoder_attention_mask=image_attn
                                             )

        # Return captions
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len('a picture of '):])
        return captions

    # Get image feature (returns torch.Size([number of images, 256])
    def get_image_feat(self, image):
        # Handle Image
        image = self.handle_image(image)
        # Get features
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat

    # Get text feature (returns torch.Size([number of images, 256])
    def get_text_feat(self, caption):
        # Get features
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(self.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat

if __name__ == '__main__':
    # Load Model
    print("-"*120)
    print("Loading model...")
    dlip_caption = DLIPProd(name='caption', weight='../save/dlip_caption_flickr.pth', device='cpu')
    dlip_retrieval = DLIPProd(name='retrieval', weight='../save/dlip_retrieval_flickr.pth', device='cpu')
    print("Finished loading model!")

    # Generate Caption
    print("-"*120)
    print("Generating...")
    image = Image.open('../export/share/datasets/vision/flickr30k/flickr30k-images/flickr30k_images/148284.jpg')
    caption = dlip_caption.generate(image)
    print("Generated Caption: {}".format(caption))
    print("Finished generating captions!")

    # Get Image Feat and Text Feat
    print("-"*120)
    print("Getting image and text feature...")
    image_feat = dlip_retrieval.get_image_feat(image)
    text_feat = dlip_retrieval.get_text_feat(caption)
    print("Image feature shape: {}".format(image_feat.shape))
    print("Text feature shape: {}".format(text_feat.shape))
    print("Finished getting features!")
    print("-"*120)
