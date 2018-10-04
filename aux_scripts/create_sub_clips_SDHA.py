from __future__ import division


from train_faster import main_run_faster
import os

from tqdm import tqdm
from utils.utils_video import get_clip_meta, skip_first_frames, filter_bb
import cv2
import numpy as np



os.system("ls -l")

vid_dir = '/hdd/SDHA2010/Output_Videos/'
vid_dir_new = '/hdd/SDHA2010/Output_Videos_cropped/'
example_size = 13
step_size = 7
new_width = 150
new_height = 150

actions_dict = {1 : 'pointing',
                2 : 'standing',
                3 : 'digging',
                4 : 'walking',
                5 : 'carrying',
                6 : 'running',
                7 : 'waving1',
                8 : 'waving2',
                9 : 'jumping'}

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

            _, res = main_run_faster()
            bb_man = filter_bb(res)

            if np.all([x==0 for x in bb_man]):
                break

            video_writers = []
            center_points = []

            for bb_ind, bb in enumerate(bb_man):

                # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writers.append(cv2.VideoWriter(new_path[:-4] + '_' + str(bb_ind) + '.avi', fourcc, fps,
                                                     (new_width, new_height)))


                center_point[0] = (bb[1] + bb[3]) // 2
                center_point[1] = (bb[0] + bb[2]) // 2

                if (center_point[0]-(new_height // 2)) < 0:
                    center_point[0] = (new_height // 2)
                elif (center_point[0]+(new_height // 2)) > frame_height:
                    center_point[0] = frame_height - (new_height // 2)

                if (center_point[1] - (new_width // 2)) < 0:
                    center_point[1] = (new_width // 2)
                elif (center_point[1] + (new_width // 2)) > frame_width:
                    center_point[1] = frame_width - (new_width // 2)

                center_points.append(center_point)

            curr_first = False

        for ind in range(len(bb_man)):
            center_p = center_points[ind]
            out = video_writers[ind]

            new_frame = frame[int(center_p[0] - new_height // 2):int(center_p[0] + new_height // 2),
                        int(center_p[1] - new_width // 2):int(center_p[1] + new_width // 2)]
            # Write the frame into the file 'output.avi'
            out.write(new_frame)

        frame_num += 1


    capture.release()

    if not np.all([x == 0 for x in bb_man]):
        for j in range(len(bb_man)):
            video_writers[j].release()



pbar = tqdm(total=108)

for root, dirs, files in os.walk(vid_dir):
    for name in files:
        folder = actions_dict[int(name.split('_')[0])]
        path = vid_dir_new + folder
        if not os.path.exists(path):
            os.mkdir(path + '/')

        vid_in = root + name

        n_frames, _ = get_clip_meta(vid_in)

        # calc number of examples in current clip
        n_ex_inclip = ((n_frames - example_size) // step_size) + 1

        if n_ex_inclip < 0:
            n_ex_inclip = 0

        for i in range(n_ex_inclip):
            vid_out = path + '/' + folder + '_' + name[:-4] + '_' + str(i) + '.avi'

            write_movie(vid_in, vid_out, i * step_size)

            print(vid_in)
            print(vid_out)

        pbar.update(1)

pbar.close()




#print(main_run_faster())




