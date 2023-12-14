import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import csv
import shutil

def get_bbox(file_name):
    with open("data/bboxes/CropOrWeed2/" + file_name + ".csv", 'r') as f:
        reader = csv.reader(f)
        # Converti ogni riga in un dizionario
        objects = []
        for row in reader:
            objects.append({
                "left": row[0],
                "top": row[1],
                "right": row[2],
                "bottom": row[3],
                "class": row[4],
                "stem_x": row[5],
                "stem_y": row[6],
            })
    return objects


def get_params(file_name):
    with open("data/params/" + file_name + ".csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        objects = []
        for row in reader:
            objects.append({
                "moisture": row[0],
                "soil": row[1],
                "lighting": row[2],
                "separability": row[3]
            })
    return objects


def create_json(file_name):
    file = {
        "file": file_name,
        "filename": 'data/images/' + file_name + '.jpg',
        "width": 1920,
        "height": 1088,
        "objects": [],
        "params": []}
    bbox = get_bbox(file_name)
    params = get_params(file_name)
    file['objects'].append(bbox)
    file['params'].append(params)
    return file


def save_img(dict):
    # "data/yaml/images/train"
    pass

#1088 1090
def save_txt(dict, file_name):
    path = "data/yaml/labels/train"
    string = []
    # print('dict')
    # print(dict)
    dict_list = dict['objects'][0]
    for data in dict_list:
        print(data)
        left = data["left"]
        top = data["top"]
        right = data["right"]
        bottom = data["bottom"]
        class_label = data["class"]
        string.append(str(class_label + " " + left + " " + right + " " + top + " " + bottom+"\n"))
    with open("data/yaml/labels/train/"+file_name + ".txt", "w") as f:
        print(string[0])
        f.writelines(string)
def copy_img(json_dict):
    path = json_dict["file"]
    origin = json_dict["filename"]
    print(path)
    shutil.copy(origin, "data/yaml/images/train/"+path+".jpg")
    pass

def create_json_dump(dataset_list, type):
    obj_list = []
    for file_name in dataset_list[:4]:
        print(file_name)
        json_dict = create_json(file_name)
        obj_list.append(json_dict)
        save_img(json_dict)
        save_txt(json_dict, file_name)
        copy_img(json_dict)


#     └── data/yaml
#            └── images
#                     └── train
#                     └── val
#            └── label
#                     └── train
#                     └── val
if __name__ == '__main__':
    with open("test_split.txt", 'r') as file:
        test_list = [line.strip() for line in file.readlines()]
    create_json_dump(test_list, "test")

    # with open("train_split.txt", 'r') as file:
    #     train_list = [line.strip() for line in file.readlines()]
    # create_json_dump(train_list, "train")
