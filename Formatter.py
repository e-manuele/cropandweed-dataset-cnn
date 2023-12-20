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


'''
prende i valori presenti nel dataset e li ritorna in formato 
dataset per yolo stile COCO
'''


def normalized_coords(x_min, y_min, y_max, x_max, stem_x, stem_y, file_name):
    image_width = 1920
    image_height = 1088
    # Calcola le coordinate centrali.
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Normalizza le coordinate
    stem_x = center_x / image_width
    stem_y = center_y / image_height
    width = width / image_width
    height = height / image_height
    label = str(stem_x) + " " + str(stem_y) + " " + str(width) + " " + str(height) + "\n"
    for el in [stem_x, stem_y, width, height]:
        if el > 1:
            print("ERROR LABELS FILENAME " + file_name + " " + label)
    return label


def save_txt_labels(dict, file_name, type_dataset):
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
        to_append = normalized_coords(int(left), int(top), int(bottom), int(right), int(stem_x), int(stem_y), file_name)
        string.append(class_label + " " + to_append)
    with open("data/yaml/labels/" + type_dataset + "/" + file_name + ".txt", "w") as f:
        f.writelines(string)


def copy_img(json_dict, file_name, type_dataset):
    path = json_dict["file"]
    origin = json_dict["filename"]
    shutil.copy(origin, "data/yaml/images/" + type_dataset + "/" + path + ".jpg")
    pass


'''
prende in input la lista dei file_name senza estensione e il tipo di dataset [train, val, test]
e salva nella relativa cartella le labels e le immagini
'''
def format_dataset(dataset_file_list, type_dataset):
    completed_files = 0
    total_files = len(dataset_file_list)

    for file_name in dataset_file_list:
        json_dict = create_json(file_name)
        save_txt_labels(json_dict, file_name, type_dataset)
        copy_img(json_dict, file_name, type_dataset)
        completed_files += 1

        percent_completed = round(completed_files / total_files * 100)
        if percent_completed %5==0:
            print("Percentage " + type_dataset+" completed:", percent_completed, "%")


if __name__ == '__main__':
    with open("materials/test_split.txt", 'r') as file:
        test_list = [line.strip() for line in file.readlines()]
    format_dataset(test_list, "test")
    with open("materials/train_split.txt", 'r') as file:
        train_list = [line.strip() for line in file.readlines()]
    length = int(len(train_list) * 0.8)
    format_dataset(train_list[:length], "train")
    format_dataset(train_list[length:], "val")

    print("Formatting done!")

#     └── data/yaml
#            └── images
#                     └── train
#                     └── val
#                     └── test
#            └── label
#                     └── train
#                     └── val
#                     └── test
