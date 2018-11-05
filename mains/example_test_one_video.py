import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from testers.example_tester_one_video import ExampleTesterOneSeq
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import sys
import argparse
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


def main():
    clip_full_path = '/hdd/UCF-ARG/rooftop_clips_stabilized/waving/person08_04_rooftop_waving.avi'
    #clip_full_path = '/hdd/SDHA2010/Output_Videos/8_10.avi'
    #clip_full_path = '/hdd/UCF-ARG/rooftop_clips_stabilized/walking/person02_03_rooftop_walking.avi'
    #clip_full_path = '/hdd/IR-new/20181015_123502_0.mp4'
    #clip_full_path = '/hdd/IR-new/Stabilized/1_3_2.mp4'
    # testing mode:
    # capture the config path from a finished experiment
    config_filename = '/hdd/test_models/exp37/configs_file.json'

    paths_filename = '/home/ADAMGE/action_recognition/action_recognition_v1/configs/paths.json'
    config = process_config(config_filename, paths_filename, 'N')

    # first configure faster arguments
    args, MODEL = faster_config()

    # create faster predictor
    pred = OfflinePredictor(PredictConfig(
        model=MODEL,
        session_init=get_model_loader(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1]))

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    #sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess_config = tf.ConfigProto()

    # create tensorflow session
    sess = tf.Session(config=sess_config)

    # create an instance of the model you want
    model = ExampleModel(config)

    # create trainer and pass all the previous components to it
    tester = ExampleTesterOneSeq(sess, model, pred, clip_full_path, config)

    # restore mobile net
    model.restore_mobile_net(sess)

    # load model
    model.load(sess)

    # testing
    tester.test()


if __name__ == '__main__':
    main()
