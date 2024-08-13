import json

if __name__ == "__main__":
    # *****************************************************************************************************
    # *****************************************************************************************************
    # Read in train.json
    file_path = '../export/share/datasets/vision/textcaps/train.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Prepare the new JSON structure
    output_data = []
    for item in data['data']:
        image_id = item['image_id']
        # For each reference_strs, create a new entry in output_data
        for caption in item['reference_strs']:
            output_data.append({
                "image": f"train_images/{item['image_path'].split('/')[-1]}",
                "caption": caption,
                "image_id": image_id
            })

    # Save train.json
    with open('../export/share/datasets/vision/textcaps/train_process.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    # *****************************************************************************************************
    # *****************************************************************************************************
    # Read in val.json
    file_path = '../export/share/datasets/vision/textcaps/val.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Prepare the new format
    new_data = []

    for item in data['data']:
        # Extract relevant data for each entry
        image_id = item['image_id']
        captions = item['reference_strs']
        image_path = f"train_images/{item['image_path'].split('/')[-1]}"

        # Prepare the entry for the new format
        entry = {
            "image": image_path,
            "caption": captions
        }
        new_data.append(entry)

    # Save the new data to a JSON file
    with open('../export/share/datasets/vision/textcaps/val_process.json', 'w') as file:
        json.dump(new_data, file, indent=4)
