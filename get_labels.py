import torch
import os
import cv2
import h5py
import json

DATA_DIR = "data"
OUTPUT_DIR = "output"

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
DEBUG_DIR = os.path.join(OUTPUT_DIR, 'debug')


def load_images_from_directory(directory):
    file_names = [f for f in os.listdir(directory) if f.endswith(".png")]

    images = []
    for filename in file_names:
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        images.append((filename, image))
    return images

def read_field(f, field):
    values = []
    if field.shape[0] == 1:
        return [field[0][0]]
    for i in range(field.shape[0]):
        ref = field[i][0]
        values.append(f[ref][:][0][0])
    return values

def read_names(f, i):
    name_ref = f["digitStruct"]["name"][i][0]
    name = ''.join([chr(c[0]) for c in f[name_ref][:]])
    return name

def read_bboxes(f, i):
    bbox = f[f["digitStruct"]["bbox"][i][0]] #Get the reference to the bbox metadata for the i-th image
    heights = read_field(f, bbox['height'])
    labels = read_field(f, bbox['label'])
    tops = read_field(f, bbox['top'])
    lefts = read_field(f, bbox['left'])
    width = read_field(f, bbox['width'])

    boxes = []
    for j in range(len(labels)):
        label = int(labels[j])
        if label == 10: label = 0 # Convert label 10 to 0
        box = {
            "height": int(heights[j]),
            "label": label,
            "top": int(tops[j]),
            "left": int(lefts[j]),
            "width": int(width[j])
        }
        boxes.append(box)

    return boxes

def save_labels(labels, output_path):
    with open(output_path, 'w') as out:
        json.dump(labels, out)


def create_label_class(path, output_path):
    digitStruct = os.path.join(path, "digitStruct.mat")
    f = h5py.File(digitStruct, 'r')
    labels = {}

    count = f['digitStruct']['bbox'].shape[0]
    for i in range(count):
        name = read_names(f, i)
        labels[name] = read_bboxes(f, i)

    save_labels(labels, output_path)

    
if __name__ == "__main__":
    images = load_images_from_directory(TRAIN_DIR)
    print(len(images))
    name, img = images[0]
    print(img.shape)       

    math_path = os.path.join(TRAIN_DIR, "digitStruct.mat")
    f = h5py.File(math_path, 'r')

    #print(f.keys())
    #print(f['digitStruct'].keys())

    #print(f['digitStruct']['name'].shape)
    #print(f['digitStruct']['bbox'].shape)

    bbox_ref = f["digitStruct"]["bbox"][0][0] # Get the first bbox reference metadata
    bbox = f[bbox_ref] #Insert the metadata reference into the file to get the actual metadata
    
    #print(bbox.keys())
    """print(bbox['height'].shape)"""

    #names = read_names(f, 0)
    #boxes = read_bboxes(f, 0)

    """vis = img.copy()
    for box in boxes:
        x0, y0 = box["left"], box["top"]
        x1, y1 = x0 + box["width"], y0 + box["height"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, str(box["label"]), (x0, max(y0 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(DEBUG_DIR, "boxes_" + name), vis)"""

    #create_label_class(TRAIN_DIR, os.path.join(OUTPUT_DIR, "labels.json"))
    create_label_class(TEST_DIR, os.path.join(DATA_DIR, "processed", "boxes_test.json"))
    create_label_class(TRAIN_DIR, os.path.join(DATA_DIR, "processed", "boxes_train.json"))



