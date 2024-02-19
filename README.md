# DLIP V2 (Distilling BLIP)

This repo replicates the performance achieved in this [paper](https://arxiv.org/abs/2308.12956).

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

## Preprocessing datasets (for pretraining)
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

Once everything is set, run json_dataset.py, and it will output a json containing all image-caption dictionary information in a new directory 'dataloader'. This dictionary will be used for pretraining using a Pytorch dataloader.
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

## Pretraining
To pretrain, run pretrain_dlip.py or type this command for multi-gpu training:
python -m torch.distributed.run --nproc_per_node=4 pretrain_dlip.py

*Note: I Utilized 4 A100 GPUs to Pretrain DLIP on CC3M, COCO, SBU, and VGO (which achieved results similar to paper).*

## Finetuning
After pretraining dlip, you can finetune the pretrained dlip model for retrieval and captioning by running train_dlip_retrieval_flickr.py and train_dlip_caption_flickr.py. The model will be finetuned on Flickr30K.

## Evaluting
If you would like to evaluate the model, run the eval scripts in the eval directory. Ensure that your are using the correct model checkpiont.
