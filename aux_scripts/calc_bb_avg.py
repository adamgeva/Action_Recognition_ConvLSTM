from __future__ import division


from train_faster import main_run_faster
import os

from tqdm import tqdm
from utils.utils_video import get_clip_meta, skip_first_frames, filter_bb
import cv2
import numpy as np
from train_faster import *


def faster_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS',
                        default='/home/ADAMGE/action_recognition/models/COCO-R50C4-MaskRCNN-Standard.npz')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation on COCO. "
                                           "This argument is the path to the output json evaluation file", default='None')
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", default='/hdd/temp.jpg')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        default=['MODE_MASK=False',
                                 'BACKBONE.WEIGHTS=/home/ADAMGE/action_recognition/models/ImageNet-ResNet50.npz'],
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN training if you're unlucky.")

    args = parser.parse_args()

    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    return args, MODEL


def get_ff_bb(orig_path, faster_pred):
    bit = 0
    capture = cv2.VideoCapture(orig_path)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    fps = capture.get(cv2.CAP_PROP_FPS)

    flag, frame = capture.read()

    # run faster to detect region every frame
    res = predict(faster_pred, frame)

    # get segments block of previous 13 frames - bb interpolation
    human_bb = filter_bb(res)

    human_bb = human_bb[0]

    if not isinstance(human_bb, int):

        width = human_bb[2] - human_bb[0]
        height = human_bb[3] - human_bb[1]

        capture.release()

        return True, (width, height)
    else:
        return False, (0, 0)



os.system("ls -l")

vid_dir = '/hdd/Combined/'
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

# first configure faster arguments
args, MODEL = faster_config()

# create faster predictor
pred = OfflinePredictor(PredictConfig(
    model=MODEL,
    session_init=get_model_loader(args.load),
    input_names=MODEL.get_inference_tensor_names()[0],
    output_names=MODEL.get_inference_tensor_names()[1]))

w_vec = []
h_vec = []
for root_o, dirs, _ in os.walk(vid_dir):
    for dir in dirs:
        for root, _, files in os.walk(root_o + dir):
            for name in files:

                vid_in = root + '/' + name

                success, (w,h) = get_ff_bb(vid_in, pred)
                if success:
                    w_vec.append(w)
                    h_vec.append(h)

                print(vid_in)

print('hello')










