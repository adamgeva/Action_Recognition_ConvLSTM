import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = process_config('/home/ADAMGE/action_recognition/action_recognition_v1/configs/example.json')

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    # create tensorflow session
    sess = tf.Session(config=sess_config)

    # create an instance of the model you want
    model = ExampleModel(config)

    # create your data generator
    data_train = DataGenerator(model, config, sess, 'train')
    data_validate = DataGenerator(model, config, sess, 'test')

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data_train, data_validate, config, logger)

    #load model if exists
    #model.load(sess)

    # training
    trainer.train()

    # testing


if __name__ == '__main__':
    main()
