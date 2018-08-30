import tensorflow as tf
from os import listdir


class BaseModel:
    def __init__(self, config):
        self.config = config

        # Attributes needed for global_step and global_epoch
        self.cur_epoch_tensor = None
        self.increment_cur_epoch_tensor = None
        self.global_step_tensor = None
        self.increment_global_step_tensor = None

        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

        # save attribute .. NOTE DON'T FORGET TO CONSTRUCT THE SAVER ON YOUR MODEL
        self.saver = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        # find latest checkpoint
        # load
        model_name = self.get_latest_model_name()
        print("Loading model checkpoint {} ...\n".format(model_name))
        self.lstm_saver.restore(sess, model_name)
        print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            # this operator if you wanna increment the global_step_tensor by yourself instead of incrementing it
            # by .minimize function in the optimizers of tensorflow
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_global_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def get_latest_model_name(self):
        list_of_files = listdir(self.config.checkpoint_dir)
        list_num = [file.split('.')[0][1:] for file in list_of_files]
        only_nums = sorted(list_num)[:-1]
        only_nums.sort(key=int)
        model_num = only_nums[-1]
        model_name = self.config.checkpoint_dir + '-' + str(model_num)
        return model_name
