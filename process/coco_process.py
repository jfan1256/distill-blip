import json

if __name__ == "__main__":
    # *****************************************************************************************************
    # *****************************************************************************************************
    # Load the input JSON data
    file_path = '../export/share/datasets/vision/coco/annotations_trainval2014/annotations/captions_train2014.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Dictionary to map image IDs to their corresponding file names
    image_files_map = {img['id']: img['file_name'] for img in data['images']}

    # List to hold the final formatted data for train.json
    formatted_data = []

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        # Check if the image_id has a valid file_name to include
        if image_id in image_files_map:
            entry = {
                "image": f"train2014/train2014/{image_files_map[image_id]}",
                "caption": caption,
                "image_id": image_id
            }
            formatted_data.append(entry)

    # Save the formatted data to a new JSON file
    with open('../export/share/datasets/vision/coco/train_process.json', 'w') as outfile:
        json.dump(formatted_data, outfile, indent=4)

    # *****************************************************************************************************
    # *****************************************************************************************************
    # Load the input JSON data
    file_path = '../export/share/datasets/vision/coco/annotations_trainval2014/annotations/captions_val2014.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a dictionary to map image IDs to their corresponding file names
    image_files_map = {img['id']: img['file_name'] for img in data['images']}

    # Dictionary to map image IDs to captions
    image_captions_map = {}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        # Check if the image_id is in the images section
        if image_id in image_files_map:
            if image_id in image_captions_map:
                image_captions_map[image_id].append(caption)
            else:
                image_captions_map[image_id] = [caption]

    # Create the list in the format required by val.json
    formatted_data = []
    for image_id, captions in image_captions_map.items():
        if image_id in image_files_map:
            formatted_entry = {
                "image": f"val2014/val2014/{image_files_map[image_id]}",
                "caption": captions
            }
            formatted_data.append(formatted_entry)

    # Save the formatted data to a new JSON file
    with open('../export/share/datasets/vision/coco/val_process.json', 'w') as outfile:
        json.dump(formatted_data, outfile, indent=4)
