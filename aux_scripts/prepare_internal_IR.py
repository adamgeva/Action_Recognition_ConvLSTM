import cv2
import os
import numpy as np

width = 383
height = 240
fps = 30


path = '/hdd/IR/'

def normalize_im(im):

    im_new = im[:,1:]
    im_new = im_new.astype(np.float64)

    minimum = im_new.min()

    maximum = im_new.max()

    im_norm = 255 * (im_new - minimum)/(maximum-minimum)

    im_norm = im_norm.astype(np.uint8)

    return im_norm

def sorting_fun(file):
    if file[-3:] == 'par':
        return 10000000
    else:
        return int(file.split('_')[-1][:-4])



subdirs = [x[0] for x in os.walk(path)]
print(subdirs)


for sub_dir in subdirs[1:]:

    path_out = sub_dir + '.avi'
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_out, fourcc, fps,
                          (width, height))


    for root, dirs, files in os.walk(sub_dir):

        base_filename = '_'.join(files[0].split('_')[:-1])

        for file_name in sorted(files, key=sorting_fun):
            if file_name[-3:] == 'par':
                continue
            im_name = sub_dir + '/' + file_name
            frame = cv2.imread(im_name, cv2.IMREAD_ANYDEPTH)
            norm_frame = normalize_im(frame)
            b = np.repeat(norm_frame[:, :, np.newaxis], 3, axis=2)
            #cv2.imshow('test', norm_frame)
            #cv2.waitKey(0)
            out.write(b)

    out.release()




