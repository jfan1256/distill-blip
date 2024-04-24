import torch
import yaml

from models.dlip_caption import dlip_caption
from models.dlip_retrieval import dlip_retrieval
from prod.dlip_image import DLIPImage
from prod.dlip_text import DLIPText

print("Saving DLIP Caption")
pth = torch.load('../save/dlip_caption_flickr_6.pth', map_location='cpu')
torch.save({'model':pth['model']}, '../save/dlip_caption_flickr.pth')
del pth

print("Saving DLIP Retrieval")
pth = torch.load('../save/dlip_retrieval_flickr_6.pth', map_location='cpu')
torch.save({'model':pth['model']}, '../save/dlip_retrieval_flickr.pth')
del pth