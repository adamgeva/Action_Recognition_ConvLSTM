import random
import re


def read_classes(class_path):

    # read classes and creates the classes dictionary
    g = open(class_path, "r")
    labels = sorted(g.readlines())
    nums = []
    names = []
    for label in labels:
        a = label.split(" ")
        nums.append(int(a[0]))
        names.append(a[2][:-1])
    label_dict = dict(zip(names, nums))
    print(label_dict)

    return label_dict


def read_data(list_path, label_dict, is_shuffle, data):
    # given path of lines of files - reads the lines, parse the class and returns both
    # Also shuffles the data if this is the train set

    # read train file
    lines = []
    f = open(list_path, "r")
    lines_ = f.readlines()
    for i in range(len(lines_)):
        lines.append(lines_[i])
    len_lines = len(lines)
    print("successfully,len(list)", len_lines)

    if is_shuffle:
        # shuffle training list
        random.shuffle(lines)

    # read train labels
    labels = []
    for video in lines:
        if data == "UCF":
            x = video.split(" ")[0].split('_')[-1][:-4]
            video_class = re.split('(\d+)', x)[0]
            ground_label = label_dict[video_class]
        elif data == "SDHA":
            video_class = video.split('_')[0]
            ground_label = label_dict[video_class]
        elif data == "Combined":
            video_class = video.split('_')[0]
            ground_label = label_dict[video_class]
        elif data == "HMDB":
            ground_label = int(video.split(' ')[1][:-1])

        labels.append(ground_label)

    print(labels)

    return lines, labels, len_lines

