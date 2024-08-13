import json

def concat_jsons(output_path, *json_files):
    concatenated_data = []
    for file_path in json_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            concatenated_data.extend(data)

    with open(output_path, 'w') as outfile:
        json.dump(concatenated_data, outfile, indent=4)

if __name__ == "__main__":
    output_path = '../export/share/datasets/vision/finetune'
    flickr30k_train = '../train/annotation/flickr30k_train.json'
    flickr30k_val = '../train/annotation/flickr30k_val.json'
    flickr30k_test = '../train/annotation/flickr30k_test.json'
    textcaps_train = '../export/share/datasets/vision/textcaps/train_process.json'
    textcaps_test = '../export/share/datasets/vision/textcaps/val_process.json'
    coco_train = '../export/share/datasets/vision/coco/train_process.json'
    coco_test = '../export/share/datasets/vision/coco/val_process.json'
    concat_jsons(output_path + '/finetune_train.json', flickr30k_train, textcaps_train, coco_train)
    concat_jsons(output_path + '/finetune_test.json', flickr30k_test, textcaps_test, coco_test)
    with open(flickr30k_val, 'r') as file:
        data = json.load(file)
    with open(output_path + '/finetune_val.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
