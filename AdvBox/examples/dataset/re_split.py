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


def calculate_split_info(path: str, rate: float = 0.2):
    train_data, train_label = read_csv_classes(path, "train.csv")
    val_data, val_label = read_csv_classes(path, "val.csv")
    test_data, test_label = read_csv_classes(path, "test.csv")

    # Union operation
    labels = (train_label | val_label | test_label)
    labels = list(labels)
    labels.sort()
    print("all classes: {}".format(len(labels)))

    re_split_labels = {}
    for i, label in enumerate(labels):
        re_split_labels[label] = i

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
    return new_train_data, new_val_data, re_split_labels

def read_label(images_dict, image_dir, label_info_list):

    labelinfo = []
    for info in label_info_list.itertuples():
        image_filename = info[1]
        image_label = info[2].strip()

        if images_dict.get(image_filename, False):
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
    # dataset_dir = '/Path to Your Dataset/'
    # image_dir = '/Path to Your Images/'
    # train_save_path = '/Place to Save Cached TrainSet/'
    # test_save_path = '/Place to Save Cached TestSet/'
    dataset_dir = './mini-imagenet'

    image_dir = os.path.join(dataset_dir, 'images')

    train_save_path = os.path.join(dataset_dir, 're_split_mini-imagenet-cache-train.pkl')
    test_save_path = os.path.join(dataset_dir, 're_split_mini-imagenet-cache-test.pkl')
    re_split_labels_save_path = os.path.join(dataset_dir, 're_split_mini-imagenet_labels.txt')

    train, test, re_split_labels = calculate_split_info(dataset_dir)

    with open(re_split_labels_save_path, 'w') as fobj:
        json.dump(re_split_labels, fobj)

    images_dict = {}
    for image in os.listdir(image_dir):
        images_dict[image] = True
    train_info = read_label(images_dict, image_dir, train)
    test_info = read_label(images_dict, image_dir, test)

    load_image(train_info, train_save_path)
    load_image(test_info, test_save_path)
