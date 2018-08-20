import random
import cv2


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
    curr_video_class = video_name[2:-6]
    video_dir = video_name[:-3]
    curr_video_full_path = ucf_path + curr_video_class + '/' + video_dir + '/' + video_name + '.mpg'

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
