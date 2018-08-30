import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import sys


def main():

    new_exp = input("Is this a new experiment? [Y/N]")
    if new_exp == 'Y':
        # capture the config path from the run arguments
        # then process the json configuration file
        config_filename = '/home/ADAMGE/action_recognition/action_recognition_v1/configs/params.json'

    elif new_exp == 'N':
        config_filename = input("Enter the full path of the config file in the old experiment folder")
    else:
        print("Wrong input")
        exit()


    paths_filename = '/home/ADAMGE/action_recognition/action_recognition_v1/configs/paths.json'
    config = process_config(config_filename, paths_filename, new_exp)


    # create the experiments dirs and write the JSON file to the dir
    create_dirs([config.summary_dir, config.checkpoint_dir], config_filename)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    # create tensorflow session
    sess = tf.Session(config=sess_config)

    # create an instance of the model you want
    model = ExampleModel(config)

    # create your data generator
    data_train = DataGenerator(model, config, sess, 'train', shuffle=True)
    data_validate = DataGenerator(model, config, sess, 'validate', shuffle=False)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data_train, data_validate, config, logger)

    # restore mobile net
    model.restore_mobile_net(sess)

    # load model if exists
    if new_exp == 'N':
        model.load(sess)

    # training
    trainer.train()

    # testing


if __name__ == '__main__':
    main()
