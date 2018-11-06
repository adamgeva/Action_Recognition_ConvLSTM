import os
from random import shuffle
import re

data = 'Combined'

# create the file with statistics
def lists_to_file(text_filename, lists, labels):
    text_file = open(text_filename, "w")

    for class_list in lists:
        for file_name in class_list:
            if data == 'Combined':

                class_name = file_name.split('_')[0]
            else:
                x = file_name.split('_')[-1][:-4]
                class_name = re.split('(\d+)', x)[0]

            class_id = labels[class_name]
            text_file.write("%s  %d\n" % (file_name, class_id))

    text_file.close()


# script for creating the data reference files:
train_percent = 100 / 100 # change below 1 only for debugging mode.
test_percent = 0 / 100
val_percent = 25 / 100


# first create the label dictionary
labels = dict()
labels = {
    'digging': 0,
    'running': 1,
    'throwing': 2,
    'walking': 3,
    'waving': 4
}

root_dir = '/hdd/IR-new/Stabilized_cropped/'

subdirs = [x[0] for x in os.walk(root_dir)]
print(subdirs)

class_lists_train = []
class_lists_val = []
class_lists_test = []

for sub_dir in subdirs[1:]:
    # create the train list
    files_train = os.listdir(sub_dir)
    # shuffle
    shuffle(files_train)

    # these two lines are for debugging purposes when we want to create a very small and fast dataset
    total_examples_in_class = int(len(files_train) * train_percent)
    files_train = files_train[:total_examples_in_class]

    # create the validation list
    files_val = []
    # we take the top 15 percent after shuffling
    for example_ind in range(int(total_examples_in_class * val_percent)):
        file_name = files_train[example_ind]
        # add to val list
        files_val.append(file_name)
        # remove from train list
        files_train.remove(file_name)

    # create the test list
    files_test = []
    # we take the top 15 percent after shuffling
    for example_ind in range(int(total_examples_in_class * test_percent)):
        file_name = files_train[example_ind]
        # add to val list
        files_test.append(file_name)
        # remove from train list
        files_train.remove(file_name)

    class_lists_train.append(files_train)
    class_lists_val.append(files_val)
    class_lists_test.append(files_test)


# print all lists to files
# train
lists_to_file(root_dir + "train_list.txt", class_lists_train, labels)
lists_to_file(root_dir + "val_list.txt", class_lists_val, labels)
lists_to_file(root_dir + "test_list.txt", class_lists_test, labels)

# print the labels file
text_file = open(root_dir + "classesInd.txt", "w")
for class_name in sorted(labels):
    text_file.write("%d  %s\n" % (labels[class_name], class_name))
text_file.close()

