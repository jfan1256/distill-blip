import pickle
import torch
import pandas as pd

from tqdm import tqdm

from prod.dlip_text import DLIPText

# Get text feature
def get_tags(model, filename, batch_size, output):
    # Get file
    with open(filename, 'r') as file:
        tags = file.read().splitlines()

    embeddings  = {}
    batch_tags = []

    # Iterate through tags
    for i in tqdm(range(len(tags)), desc='Getting tag features'):
        # Extend tag to be 'a picture of ' + tag
        tag = 'a picture of ' + tags[i]
        batch_tags.append(tag)

        # Get text features for modified tag in batches
        if len(batch_tags) == batch_size or i == len(tags) - 1:
            # Fetch features for the batch of tags
            batch_features = model.get_text_feat(batch_tags)
            # Store the features in the dictionary with the original tag as key
            for original_tag, feat in zip(batch_tags, batch_features):
                feat = feat.detach().cpu().numpy()
                embeddings[original_tag] = feat
            # Clear batch_tags for next batch
            batch_tags = []

    # Specify the output filename for the current batch
    output_filename = f"{output}.p"
    with open(output_filename, 'wb') as f:
        pickle.dump(embeddings, f)

if __name__ == '__main__':
    # Load model
    print("-"*120)
    print("Loading model...")
    dlip_text = DLIPText(weight='../save/dlip_retrieval_text_flickr.pth', device='cuda')
    print("Finished loading model.")

    # Get features
    print("-"*120)
    print("Getting genre.txt features...")
    get_tags(dlip_text, '../tags/genre.txt', 16, '../save/genre')
    print("-"*120)
    print("Getting tags.txt features..")
    get_tags(dlip_text, '../tags/tags.txt', 16, '../save/tags')
    print("-"*120)
    print("Getting kinetics.txt features...")
    get_tags(dlip_text, '../tags/kinetics.txt', 16, '../save/kinetics')