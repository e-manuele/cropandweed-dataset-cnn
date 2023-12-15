import csv
import shutil


def get_bbox(file_name):
    with open("data/bboxes/CropOrWeed2/" + file_name + ".csv", 'r') as f:
        reader = csv.reader(f)
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


def normalized_coords(x_min, y_min, y_max, x_max, stem_x, stem_y):
    image_width = 1920
    image_height = 1088
    # Calcola le coordinate centrali.
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Normalizza le coordinate. round(val, 6)
    stem_x = round(stem_x / image_width, 6)
    stem_y = round(stem_y / image_height, 6)
    width = round(width / image_width, 6)
    height = round(height / image_height, 6)
    label = str(stem_x) + " " + str(stem_y) + " " + str(width) + " " + str(height) + "\n"
    return label


def save_txt_labels(dict, file_name, type_dataset):
    path = "data/yaml/labels/train"
    string = []
    dict_list = dict['objects'][0]
    for data in dict_list:
        left = data["left"]
        top = data["top"]
        right = data["right"]
        bottom = data["bottom"]
        class_label = data["class"]
        stem_x = data["stem_x"]
        stem_y = data["stem_y"]
        to_append = normalized_coords(int(left), int(top), int(bottom), int(right), int(stem_x), int(stem_y))
        string.append(class_label + " " + to_append)
    with open("data/yaml/labels/" + type_dataset + "/" + file_name + ".txt", "w") as f:
        f.writelines(string)


def copy_img(json_dict, file_name, type_dataset):
    path = json_dict["file"]
    origin = json_dict["filename"]
    shutil.copy(origin, "data/yaml/images/" + type_dataset + "/" + path + ".jpg")
    pass


def format_dataset(dataset_file_list, type_dataset):
    obj_list = []

    for file_name in dataset_file_list:
        #print(file_name)
        json_dict = create_json(file_name)
        save_txt_labels(json_dict, file_name, type_dataset)
        copy_img(json_dict, file_name, type_dataset)


#     └── data/yaml
#            └── images
#                     └── train
#                     └── val
#            └── label
#                     └── train
#                     └── val

if __name__ == '__main__':
    # with open("materials/test_split.txt", 'r') as file:
    #     test_list = [line.strip() for line in file.readlines()]
    # format_dataset(test_list, "test")

    with open("materials/train_split.txt", 'r') as file:
        train_list = [line.strip() for line in file.readlines()]
    length = int(len(train_list) * 0.8)
    format_dataset(train_list[:length], "train")
    format_dataset(train_list[length:], "val")