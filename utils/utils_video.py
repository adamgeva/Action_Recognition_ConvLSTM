import random
import cv2
import re
from imgaug import augmenters as iaa
import numpy as np


# select a clip from all the training clips
def select_clip(train_set_length, train_names, data_dict):
    clip_number = random.randint(0, train_set_length - 1)
    curr_line = train_names[clip_number]
    return line_to_path(curr_line, data_dict)


# returns the number of frames in the clip
def get_clip_meta(video_path):
    capture = cv2.VideoCapture(video_path)
    num_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    frame_dim = (height, width, 3)
    capture.release()
    return num_of_frames, frame_dim


# takes a line and returns the full path and class of the vid line
def line_to_path(line, ucf_path):
    video_name = line.split(" ")[0]
    x = video_name.split('_')[-1][:-4]
    curr_video_class = re.split('(\d+)', x)[0]
    curr_video_full_path = ucf_path + curr_video_class + '/' + video_name

    return curr_video_full_path, curr_video_class


# modifies the RGB seq and size to fit the input size of mobile net
def val_reprocess(config, frame):
    # for mobile net - input value is -1 to 1
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = cv2.resize(image, dsize=tuple(config.frame_size[0:2]), interpolation=cv2.INTER_LINEAR)
    res = res.astype('float32') / 128. - 1
    #means = np.reshape(VGG_MEAN, [1, 1, 3])
    #centered_image = res - means  # (4)
    #return centered_image
    return res


# read first frames until reaching the first frame to sample from
def skip_first_frames(vid_capture, first_frame):
    frame_counter = 0
    bit = 0

    while (vid_capture.isOpened()) & (frame_counter < (first_frame - 1)) & (bit == 0):
        flag, frame = vid_capture.read()
        if flag == 0:
            print('********************ERROR reading first frames****************************')
            bit = 1
            break
        frame_counter += 1

    return bit, vid_capture


# performs the same augmentation on all frames
def augment_frames(frames):
    frames = (frames + 1.0) * 128.
    frames = frames.astype('uint8')
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.5, 2.0)),
        #iaa.Add((-20, 20)),
        #iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
        #iaa.ContrastNormalization(alpha=(0.5,1.5)),
        #iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        #iaa.Affine(rotate=(0, 0))
    ])

    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
    frames_aug = np.zeros(frames.shape)
    for i in range(frames.shape[0]):
        frame_aug = seq_det.augment_image(frames[i])
        #cv2.imshow('ImageWindow', frame_aug)
        #cv2.waitKey()
        frames_aug[i] = frame_aug
    frames_aug = frames_aug.astype('float32') / 128. - 1
    return frames_aug


def filter_bb(res):
    bb_man = []
    no_man_detection = True

    for detection in res:
        bb = detection[0]
        score = detection[1]
        id = detection[2]
        if (id == 1) and score > 0.8:  # man detected
            bb_man.append(bb)
            no_man_detection = False

    if no_man_detection:
        print('no man detection!!!')
        bb_man = [0, 0, 0, 0]

    return bb_man
