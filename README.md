# DLIP V2 (Distilling BLIP)

----------

Distilled BLIP model (VIT Small, BERT Small, and BLIP Base) achieves similar performance to BLIP with 4x the speed in captioning and retrieval. This repo replicates the performance achieved in this [paper](https://arxiv.org/abs/2308.12956).

----------

## Instructions to download datasets

To download CC3M (Google Conceptual Captions 3M), COCO, and SBU (SBU Captions) run the following commands in each subdirectory (i.e., cd to cc3m and run the script):

```
/datadrive
  ├── cc3m
  ├── mscoco
  ├── sbu
  └── vgo
```
  
Refer to Img2Dataset directly for more information [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md). 

#### CC3M
```bash
img2dataset --url_list cc3m.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format files --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 224
```
#### COCO
```bash
img2dataset --url_list mscoco.parquet --input_format "parquet" --url_col "URL" --caption_col "TEXT" --output_format files --output_folder mscoco --processes_count 16 --thread_count 64 --image_size 224
```
#### SBU 
```bash
img2dataset --url_list sbu-captions-all.json --input_format "json" --url_col "image_urls" --caption_col "captions" --output_format files --output_folder sbucaptions --processes_count 16 --thread_count 64 --image_size 224
```
#### VGO
Refer to [here](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) for more details.
#### Flickr30K
Refer to [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) for more details.

----------

## Preprocess (for pretraining)
Once everything is downloaded, to ensure that json_dataset.py works, please ensure this file structure:

```
/datadrive
  ├── cc3m
  │   └── cc3m                  <- Contains all the .parquet, 0000 subdir, etc.
  ├── mscoco
  │   └── mscoco                <- Contains all the .parquet, 0000 subdir, etc.
  ├── sbu
  │   └── sbucaptions           <- Contains all the .parquet, 0000 subdir, etc.
  ├── vgo
  │   ├── VG_100K               <- Contains all part 1 images
  │   ├── VG_100K_2             <- Contains all part 2 images - make sure to move all images in VG_100K_2 to VG_100K.
  │   └── region_descriptions.json
```

Once everything is set, run json_dataset.py, and it will output all.json in a self-created directory '/datadrive/dataloader', 
which contains all image-caption items from CC3M, COCO, SBU, and VGO . This dictionary will be used for pretraining dlip.
The output dictionary (all.json) will look like this:

```json
[{
  "caption": "bridge street in the rain ..",
  "image_root": "/datadrive/sbu/sbucaptions/00000/000000015.jpg",
  "id": 0
},
{
  "caption": "a bird perched on a tree ..",
  "image_root": "/datadrive/sbu/sbucaptions/00000/000000016.jpg",
  "id": 0
}]
```
----------
## Pretrain
To pretrain, run pretrain_dlip.py or run this command in command prompt for multi-gpu training (--nproc_per_node is the number of gpus to use):
```bash
python -m torch.distributed.run --nproc_per_node=4 pretrain_dlip.py
```
*Note: I Utilized 4 A100 GPUs to Pretrain DLIP on CC3M, COCO, SBU, and VGO.*

----------

## Finetune
After pretraining dlip, you can finetune the pretrained dlip model for retrieval and captioning by running train_dlip_retrieval_flickr.py and train_dlip_caption_flickr.py. The model will be finetuned on Flickr30K.

Here is a result comparison between DLIP Retrieval vs. BLIP (CapFilt-L) Retrieval on Flickr30K Test:
```

| Metric           | DLIP   | BLIP   |
|------------------|--------|--------|
| `train_lr`       | 0.000  | 0.000  |
| `train_loss_itm` | 0.073  | 0.054  |
| `train_loss_ita` | 2.349  | 1.968  |
| `test_txt_r1`    | 88.9   | 97.2   |
| `test_txt_r5`    | 98.4   | 99.9   |
| `test_txt_r10`   | 99.7   | 100.0  |
| `test_img_r1`    | 76.36  | 87.6   |
| `test_img_r5`    | 93.44  | 97.7   |
| `test_img_r10`   | 96.12  | 98.9   |

```
As you can see, DLIP is practically as accurate as DLIP despite being 10x smaller.
More details on BLIP results can be found [here](https://arxiv.org/pdf/2201.12086.pdf).

----------

## Evaluation
If you would like to evaluate the model, run the eval scripts in the eval directory. Ensure that you are using the correct model checkpoint.

----------

## Production
If you would like to use the model for production, utilize the DLIPProd class in the prod directory. 

----------
### Demo
```
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
```
### Output
```
------------------------------------------------------------------------------------------------------------------------
Loading model...
Finished loading model!
------------------------------------------------------------------------------------------------------------------------
Generating...
Generated Caption: ['an african american in front of the spanish door']
Finished generating captions!
------------------------------------------------------------------------------------------------------------------------
Getting image and text feature...
Image feature shape: torch.Size([1, 256])
Text feature shape: torch.Size([1, 256])
Finished getting features!
------------------------------------------------------------------------------------------------------------------------
Loading model...
Finished loading model!
------------------------------------------------------------------------------------------------------------------------
Generating...
Generated Caption: ['an african american in front of the spanish door']
Finished generating captions!
------------------------------------------------------------------------------------------------------------------------
Getting image and text feature...
Image feature shape: torch.Size([1, 256])
Text feature shape: torch.Size([1, 256])
Finished getting features!
------------------------------------------------------------------------------------------------------------------------
```

