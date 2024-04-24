import json
import os
import cv2
import pandas as pd
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from ram.models import ram_plus
from ram import inference_ram as inference

# Handle Image Cases
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

# Transform Image
def transform(image):
    # Resize the image
    image_resized = cv2.resize(np.asarray(image), (384, 384), interpolation=cv2.INTER_CUBIC)

    image_scaled = image_resized / 255.0

    output = torch.tensor(image_scaled).permute(2, 0, 1).float()
    # Normalize the image
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    image_normalized = (output - mean) / std
    return image_normalized

# Get RAM Tags in Batches
def get_tag_batch(items, model):
    with torch.no_grad():
        images = [handle_case(transform, image) for image in items]
        images = torch.stack(images)
        images = images.to(device='cuda')
        results = inference(images, model)
    return results

if __name__ == '__main__':
    # Params
    directory_path = '../export/share/datasets/vision/flickr30k/flickr30k-images/'
    output_path = '../export/share/datasets/tag/'
    batch_size = 50

    # Load Model
    ram_plus_model = ram_plus(pretrained='../save/ram_plus_swin_large_14m.pth', image_size=384, vit='swin_l')
    ram_plus_model = ram_plus_model.to(device='cuda')

    # Get image names
    image_names = os.listdir(directory_path)
    image_paths = [os.path.join(directory_path, name) for name in image_names if name.endswith(('.png', '.jpg', '.jpeg'))]
    image_ids = [os.path.splitext(name)[0] for name in image_names if name.endswith(('.png', '.jpg', '.jpeg'))]

    # Process in batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    all_tags = []
    all_ids = []

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
        batch_images = [Image.open(path) for path in batch_paths]
        batch_ids = image_ids[i * batch_size:(i + 1) * batch_size]
        batch_tags = get_tag_batch(batch_images, ram_plus_model)
        all_tags.extend(batch_tags)
        all_ids.extend(batch_ids)

    # Ensure each tag list has 10 tags, filling missing tags with None
    all_tags_padded = [tags + [None] * (10 - len(tags)) if len(tags) < 10 else tags[:10] for tags in all_tags]

    # Export dataframe
    print("Export dataframe to", output_path)
    df_columns = ['id'] + [f'tag_{i + 1}' for i in range(10)]
    df = pd.DataFrame([[id] + tags for id, tags in zip(all_ids, all_tags_padded)], columns=df_columns)
    df.to_csv(output_path + 'image_tags_10.csv', index=False)

    # Export dataframe to JSON
    df = pd.read_csv(output_path + 'image_tags_10.csv')
    print("Number of rows in the image tag dataframe before dropping NAN: ", len(df))
    df = df.dropna()
    print("Number of rows in the image tag dataframe after dropping NAN: ", len(df))
    json_records = []
    current_image_id = 0
    last_image = None

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the image path
        image_path = f'flickr30k-images/{row["id"]}.jpg'

        # Check if this is a new image to update image_id
        if last_image != image_path:
            if last_image is not None:
                current_image_id += 1
            last_image = image_path

        # Construct the JSON record
        json_record = {
            "image": image_path,
            "tag": [tag for tag in row[1:] if isinstance(tag, str)],
            "image_id": current_image_id
        }

        # Append the JSON record to the list
        json_records.append(json_record)

    # Write the json to a file
    print("Export json to", output_path)
    with open(output_path + 'image_tags_10.json', 'w') as json_file:
        json.dump(json_records, json_file)