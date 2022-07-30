import os
import json
import numpy as np
import pandas as pd


def read_csv_classes(csv_dir: str, csv_name: str):
    data = pd.read_csv(os.path.join(csv_dir, csv_name))
    label_set = set(data["label"].drop_duplicates().values)

    print("{} have {} images and {} classes.".format(csv_name,
                                                     data.shape[0],
                                                     len(label_set)))
    return data, label_set


def calculate_split_info(path: str, label_dict: dict, rate: float = 0.2):
    train_data, train_label = read_csv_classes(path, "train.csv")
    val_data, val_label = read_csv_classes(path, "val.csv")
    test_data, test_label = read_csv_classes(path, "test.csv")

    # Union operation
    labels = (train_label | val_label | test_label)
    labels = list(labels)
    labels.sort()
    print("all classes: {}".format(len(labels)))

    # create classes_name.json
    classes_label = dict([(label, [index, label_dict[label]]) for index, label in enumerate(labels)])
    json_str = json.dumps(classes_label, indent=4)
    with open('classes_name.json', 'w') as json_file:
        json_file.write(json_str)

    # concat csv data
    data = pd.concat([train_data, val_data, test_data], axis=0)
    print("total data shape: {}".format(data.shape))

    # split data on every classes
    num_every_classes = []
    split_train_data = []
    split_val_data = []
    for label in labels:
        class_data = data[data["label"] == label]
        num_every_classes.append(class_data.shape[0])

        # shuffle
        shuffle_data = class_data.sample(frac=1, random_state=1)
        num_train_sample = int(class_data.shape[0] * (1 - rate))
        split_train_data.append(shuffle_data[:num_train_sample])
        split_val_data.append(shuffle_data[num_train_sample:])

    # concatenate data
    new_train_data = pd.concat(split_train_data, axis=0)
    new_val_data = pd.concat(split_val_data, axis=0)
    return new_train_data, new_val_data


def split(data_dir):
    label_path = os.path.join(os.path.dirname(__file__), 'mini-imagenet/mini_imagenet1000_labels.txt')
    with open(label_path) as info:
        label_dict = eval(info.read())

    return calculate_split_info(data_dir, label_dict)


def read_label(image_dir, label_info_list):
    image_filepaths = os.listdir(image_dir)

    labelinfo = []
    for info in label_info_list.itertuples():
        image_filename = info[1]
        image_label = info[2].strip()

        if image_filename in image_filepaths:
            image_filepath = os.path.join(image_dir, image_filename)
            labelinfo.append((image_filepath, image_label))

    return labelinfo


def load_image(labelinfo, save_path):
    import cv2
    import pickle
    images = []
    classes = dict()
    class_num = 0
    for i, data in enumerate(labelinfo):
        img_path, label = data
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, [84, 84])
        images.append(img)
        if label in classes:
            classes[label].append(i)
        else:
            classes[label] = [i]
            class_num += 1

    images = np.stack(images)
    content = {'image_data': images,
               'class_dict': classes}
    f = open(save_path, 'wb')
    pickle.dump(content, f)
    f.close()


if __name__ == '__main__':
    # Change to the path on your own device
    dataset_dir = '/Path to Your Dataset/'
    image_dir = '/Path to Your Images/'
    train_save_path = '/Place to Save Cached TrainSet/'
    test_save_path = '/Place to Save Cached TestSet/'

    train, test = split(dataset_dir)
    train_info = read_label(image_dir, train)
    test_info = read_label(image_dir, test)
    load_image(train_info, train_save_path)
    load_image(test_info, test_save_path)
