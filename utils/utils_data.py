import random


def read_classes(class_path):

    # read classes and creates the classes dictionary
    g = open(class_path, "r")
    labels = sorted(g.readlines())
    nums = []
    names = []
    for label in labels:
        a = label.split(" ")
        nums.append(int(a[0])-1)
        names.append(a[1][:-1])
    label_dict = dict(zip(names, nums))
    print(label_dict)

    return label_dict


def read_data(list_path, label_dict, is_shuffle):
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
        video_class = str(video.split(" ")).split("_")[1]
        ground_label = label_dict[video_class]
        labels.append(ground_label)

    print(labels)

    return lines, labels, len_lines

