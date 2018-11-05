from __future__ import division

import os

from tqdm import tqdm
from utils.utils_video import get_clip_meta, skip_first_frames, filter_bb
import cv2
import numpy as np



os.system("ls -l")

vid_dir = '/hdd/IR-new/Stabilized/'
vid_dir_new = '/hdd/IR-new/Stab_Cropped_Random/walking/'
example_size = 40
step_size = 40
new_width = 150
new_height = 150


subdirs = [x[0] for x in os.walk(vid_dir)]
print(subdirs)


def write_movie(orig_path, new_path, start_frame):

    bit = 0
    capture = cv2.VideoCapture(orig_path)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    fps = capture.get(cv2.CAP_PROP_FPS)

    # skip to starting frame
    bit, capture = skip_first_frames(capture, start_frame)

    frame_num = 0
    curr_first = True

    center_point = [0, 0]
    while (capture.isOpened()) & (frame_num < example_size) & (bit == 0):
        flag, frame = capture.read()
        if flag == 0:
            bit = 1
            print("******ERROR: Could not read frame in " + orig_path + " initial frame: "
                  + str(start_frame) + " frame_num: " + str(frame_num))
            break

        # find center point in first frame
        if curr_first:
            cv2.imwrite('/hdd/temp.jpg', frame)



            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(new_path, fourcc, fps, (new_width, new_height))


            center_point[0] = int(np.random.rand() * frame_height)
            center_point[1] = int(np.random.rand() * frame_width)

            if (center_point[0]-(new_height // 2)) < 0:
                center_point[0] = (new_height // 2)
            elif (center_point[0]+(new_height // 2)) > frame_height:
                center_point[0] = frame_height - (new_height // 2)

            if (center_point[1] - (new_width // 2)) < 0:
                center_point[1] = (new_width // 2)
            elif (center_point[1] + (new_width // 2)) > frame_width:
                center_point[1] = frame_width - (new_width // 2)


            curr_first = False

        center_p = center_point

        new_frame = frame[int(center_p[0] - new_height // 2):int(center_p[0] + new_height // 2),
                    int(center_p[1] - new_width // 2):int(center_p[1] + new_width // 2)]
        # Write the frame into the file 'output.avi'
        out.write(new_frame)

        frame_num += 1

    out.release()
    capture.release()



pbar = tqdm(total=108)
num = 0
for root, dirs, files in os.walk(vid_dir):
    for k, name in enumerate(files):

        path = vid_dir_new

        vid_in = root + name

        n_frames, _ = get_clip_meta(vid_in)

        # calc number of examples in current clip
        n_ex_inclip = ((n_frames - example_size) // step_size) + 1

        if n_ex_inclip < 0:
            n_ex_inclip = 0

        n_samples = 3
        for i in range(0, n_ex_inclip):
            for j in range(n_samples):
                vid_out = path + 'walking_' + str(num) + '.avi'
                write_movie(vid_in, vid_out, i * step_size)
                num = num + 1
            print(vid_in)
            print(vid_out)

        pbar.update(1)

pbar.close()




#print(main_run_faster())




