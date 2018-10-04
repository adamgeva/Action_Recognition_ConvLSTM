from __future__ import division


from train_faster import main_run_faster
import os

from tqdm import tqdm
from vid_util import get_clip_meta, skip_first_frames
import cv2



os.system("ls -l")

vid_dir = '/hdd/UCF-ARG/rooftop_clips_stabilized/'
vid_dir_new = '/hdd/UCF-ARG/rooftop_clips_stab_crop/'
example_size = 40
step_size = 20
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

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(new_path, fourcc, fps,
                          (new_width, new_height))

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
            bb = main_run_faster()

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
            curr_first = False


        new_frame = frame[int(center_point[0] - new_height // 2):int(center_point[0] + new_height // 2),
                    int(center_point[1] - new_width // 2):int(center_point[1] + new_width // 2)]
        # Write the frame into the file 'output.avi'
        out.write(new_frame)

        frame_num += 1


    capture.release()
    out.release()

pbar = tqdm(total=468)

for sub_dir in subdirs[1:]:
    for root, dirs, files in os.walk(sub_dir):
        folder = root.split('/')[-1]
        path = vid_dir_new + folder +'/'
        os.mkdir(path)
        for name in files:
            vid_in = root + '/' + name

            n_frames, _ = get_clip_meta(vid_in)

            # calc number of examples in current clip
            n_ex_inclip = ((n_frames - example_size) // step_size) + 1

            if n_ex_inclip < 0:
                n_ex_inclip = 0

            for i in range(n_ex_inclip):
                vid_out = path + name[:-4] + str(i) + '.avi'

                write_movie(vid_in, vid_out, i * step_size)

                print(vid_in)
                print(vid_out)

            pbar.update(1)

pbar.close()




#print(main_run_faster())




