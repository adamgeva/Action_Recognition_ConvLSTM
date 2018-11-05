from __future__ import division

import os

class_id_dict = {
    'digging' : 0,
    'running' : 1,
    'throwing'  : 2,
    'walking'  : 3,
    'waving'  : 4
}


os.system("ls -l")

vid_dir = '/hdd/SDHA2010/Output_Videos_cropped_merged/'

subdirs = [x[0] for x in os.walk(vid_dir)]
print(subdirs)

file_out = open(vid_dir + 'test_list.txt', 'w')


for sub_dir in subdirs[1:]:
    for root, dirs, files in os.walk(sub_dir):
        dir = root.split('/')[-1]
        if dir in class_id_dict:
            class_id = class_id_dict[dir]
        else:
            continue

        for file_name in files:
            file_out.write(file_name + '  ' + str(class_id) + '\n')
        #print(files)
        #print(root.split('/')[-1])

file_out.close()
