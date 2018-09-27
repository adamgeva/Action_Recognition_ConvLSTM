import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from testers.example_tester_one_video import ExampleTesterOneSeq
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import sys


def main():

    clip_full_path = ' '
    # testing mode:
    # capture the config path from a finished experiment
    config_filename = '/home/ADAMGE/action_recognition/action_recognition_v1/experiments/2018-08-27 08:39:10/configs_file.json'

    paths_filename = '/home/ADAMGE/action_recognition/action_recognition_v1/configs/paths.json'
    config = process_config(config_filename, paths_filename, 'N')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    # create tensorflow session
    sess = tf.Session(config=sess_config)

    # create an instance of the model you want
    model = ExampleModel(config)

    # create trainer and pass all the previous components to it
    tester = ExampleTesterOneSeq(sess, model, clip_full_path, config)

    # restore mobile net
    model.restore_mobile_net(sess)

    # load model
    model.load(sess)

    # testing
    tester.test()


if __name__ == '__main__':
    main()
