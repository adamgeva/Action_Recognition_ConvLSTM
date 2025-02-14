import tensorflow as tf


class BaseTest:
    def __init__(self, sess, model, config, logger):
        """
        Constructing the trainer
        :param sess: TF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all the configurations you have specified in the json
        :param logger: logger class which will summarize and write the values to the tensorboard
        :param data_loader: The data loader if specified. (You will find Dataset API example)
        """
        # Assign all class attributes
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def test(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        raise NotImplementedError


    def test_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        raise NotImplementedError
