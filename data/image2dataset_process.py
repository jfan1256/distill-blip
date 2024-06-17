import os
import json

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Process subdir (this function will be parallelized)
def process_subdir(subdir_info):
    subdir, files = subdir_info
    images_info = []
    for file in files:
        # Check if the file is a JSON file
        if file.endswith(".json"):
            file_path = os.path.join(subdir, file)
            # Open and read the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

                # Extract the caption and key
                caption = data.get("caption", "")
                key = data.get("key", "")
                image_root = os.path.join(subdir, f"{key}.jpg")

                if not caption or not key:
                    continue
                else:
                    images_info.append({"caption": caption, "image_root": image_root})

    return images_info

# Create entire json for sbu, coco, and cc3m
def dataset_json(parent_dir, output_dir, json_name):
    # List to hold all the image information
    images_info = []

    # Collect all subdirectories and their files
    subdir_files = []
    for subdir, dirs, files in os.walk(parent_dir):
        subdir_files.append((subdir, files))

    # Process each subdirectory in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        futures  = [executor.submit(process_subdir, subdir_info) for subdir_info in subdir_files]

        # Append each item
        for future in tqdm(futures, desc="Processing directories"):
            for item in future.result():
                images_info.append(item)

    # Specify the path for the output JSON file
    output_file_path = os.path.join(output_dir, f"{json_name}.json")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write the compiled information to a new JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(images_info, outfile, indent=4)

    print(f"Compiled JSON file created at {output_file_path}")

# Process image dict (this function will be parallelized)
def process_image_dict(image_dict, image_dir, image_area_threshold):
    filtered_annotations = []
    for item in image_dict['regions']:
        phrase = item.get("phrase", "")
        width = item.get("width", 0)
        height = item.get("height", 0)
        image_id = item.get("image_id", 0)
        region_area = width * height
        if len(phrase.split()) > 4 and region_area > image_area_threshold:
            image_root = os.path.join(image_dir, f"{image_id}.jpg")
            filtered_annotations.append({"caption": phrase, "image_root": image_root})
    return filtered_annotations

# Create filtered json for vgo
def dataset_vgo(json_dir, image_dir, output_dir, json_name):
    # List to hold all the image information
    images_info = []

    # Load the JSON data
    with open(json_dir, 'r') as file:
        data = json.load(file)

    # Predefined image size (800x600) and the threshold for the region size
    image_area_threshold = 800 * 600 * 0.2

    # Process each image_dict in parallel and maintain order
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_image_dict, image_dict, image_dir, image_area_threshold) for image_dict in data]

        # Append each item
        for future in tqdm(futures, desc="Processing image dicts"):
            for item in future.result():
                images_info.append(item)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Specify the path for the output JSON file
    output_file_path = os.path.join(output_dir, f"{json_name}.json")

    # Write the items to output json file
    with open(output_file_path, 'w') as outfile:
        json.dump(images_info, outfile, indent=4)

    print(f"Filtered JSON file created at {output_file_path}")

# Concat json datasets into all.json
def concat_all(parent_dir, output_dir, json_name):
    # List of json datasets
    json_datasets = ["sbu.json", "coco.json", "cc3m.json", "vgo.json"]

    # Initialize a list to hold all annotations from all datasets
    all_json = []

    # Unique ID counter
    unique_id = 0

    # Iterate over each JSON file and append its contents to `all_annotations`
    for json_filename in tqdm(json_datasets, desc="Concat all"):
        json_path = os.path.join(parent_dir, json_filename)
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                for item in data:
                    # Add a unique id to each item
                    item["id"] = unique_id
                    all_json.append(item)
                    # Increment the unique ID for the next item
                    unique_id += 1
        else:
            print(f"Warning: {json_path} does not exist and will be skipped.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write the items to json file
    final_json_path = os.path.join(output_dir, f"{json_name}.json")
    with open(final_json_path, 'w') as final_json_file:
        json.dump(all_json, final_json_file, indent=4)

    print(f"All annotations have been concatenated into {final_json_path}")

if __name__ == "__main__":
    '''
    output_dir = /datadrive/dataloader

    SBU Captions (sbu):
        parent_dir = /datadrive/sbu/sbucaptions
        json_name = sbu

    COCO (coco):
        parent_dir = /datadrive/mscoco/mscoco
        json_name = coco

    Google Conceptual Captions (cc3m):
        parent_dir = /datadrive/cc3m/cc3m
        json_name = cc3m

    Visual Genome (vgo):
        json_dir = /datadrive/vgo/region_descriptions.json
        image_dir = /datadrive/vgo/VG_100K
        json_name = vgo
    
    All (all):
        parent_dir = /datadrive/dataloader
        output_dir = /datadrive/dataloader
        json_name = all
    '''

    # SBU
    print("-" * 60 + "SBU" + "-" * (60-3))
    dataset_json(parent_dir="/datadrive/sbu/sbucaptions", output_dir="/datadrive/dataloader", json_name="sbu")

    # COCO
    print("-" * 60 + "COCO" + "-" * (60-4))
    dataset_json(parent_dir="/datadrive/mscoco/mscoco", output_dir="/datadrive/dataloader", json_name="coco")

    # CC3M
    print("-" * 60 + "cc3m" + "-" * (60-4))
    dataset_json(parent_dir="/datadrive/cc3m/cc3m", output_dir="/datadrive/dataloader", json_name="cc3m")

    # VGO
    print("-" * 60 + "VGO" + "-" * (60-3))
    dataset_vgo(json_dir="/datadrive/vgo/region_descriptions.json", image_dir="/datadrive/vgo/VG_100K", output_dir="/datadrive/dataloader", json_name="vgo")

    # ALL
    print("-" * 60 + "ALL" + "-" * (60-3))
    concat_all(parent_dir="/datadrive/dataloader", output_dir="/datadrive/dataloader", json_name="all")





