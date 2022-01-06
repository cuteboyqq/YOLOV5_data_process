import os
import json
from tqdm import tqdm
import cv2

TRAIN_LABEL_NAME = 'bdd100k_labels_images_train.json'
VAL_LABEL_NAME = 'bdd100k_labels_images_val.json'

classes = [
    "person",
    "bike",
    "car",
    "motor",
    "red ts",
    "bus",
    "green ts",
    "truck",
    "yellow ts",
    "off ts", #off ts
    "red left",
    "stop sign",
    "green straight ts",
    "green right ts",
    "red right ts",
    "green left ts",
    "rider"
]

'''
classes = [
    "traffic sign"
]
'''
counter = {}

for c in classes:
    counter[c] = 0



def Analysis_path(labelPath):
    label_file = labelPath.split("/")[-1]
    #print(label_file)
    label_type = labelPath.split("/")[-2]
    #label_type = label_type + "-add-TS"
    #print(label_type)
    new_labelPath = os.path.join(r"/home/ali/bdd100k_ori/labels/",label_type,label_file)
    if not os.path.exists(os.path.join(r"/home/ali/bdd100k_ori/labels/",label_type)):
        os.makedirs(os.path.join(r"/home/ali/bdd100k_ori/labels/",label_type))
    return new_labelPath
def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images', help="images path",
                        type=str, default='/home/ali/bdd100k/images')
    parser.add_argument('-l', '--labels', help="labels path",
                        type=str, default="/home/ali/bdd100k/labels_json")
    parser.add_argument('-t', '--type', help="type of dataset",
                        choices=['train', 'val'], default='train')

    return parser.parse_args()


def convertBdd100k2yolo(imageFileName, label):
    img = cv2.imread(imageFileName)
    width, height = img.shape[1], img.shape[0]
    dw = 1.0/width
    dh = 1.0/height

    catName = label['category']
    classIndex = classes.index(catName)
    #classIndex = 9
    roi = label['box2d']

    w = roi['x2']-roi['x1']
    h = roi['y2']-roi['y1']
    x_center = roi['x1'] + w/2
    y_center = roi['y1'] + h/2

    x_center, y_center, w, h = x_center*dw, y_center*dh, w*dw, h*dh

    return "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(classIndex, x_center, y_center, w, h)


if __name__ == '__main__':
    args = get_args()
    if args.type == 'train':
        imageRootPath = os.path.join(args.images, 'train')
        labelFilePath = os.path.join(args.labels, TRAIN_LABEL_NAME)
    else:
        imageRootPath = os.path.join(args.images, 'val')
        labelFilePath = os.path.join(args.labels, VAL_LABEL_NAME)

    with open(labelFilePath) as file:
        lines = json.load(file)
        print("loaded labels")

    for line in tqdm(lines):
        #image name
        name = line['name']
        labels = line['labels']
        imagePath = os.path.join(imageRootPath, name)
        labelPath = imagePath.replace('jpg', 'txt')
        if os.path.exists(labelPath):
            os.remove(labelPath)
        new_labelPath = Analysis_path(labelPath)
        if not os.path.isfile(os.path.realpath(imagePath)):
            continue
        with open(new_labelPath, 'w') as file:
            #go through all labels
            for label in labels:
                cat = label['category']
                if cat in classes:
                    counter[cat] += 1
                    file.write(convertBdd100k2yolo(imagePath,label))

    print(counter)