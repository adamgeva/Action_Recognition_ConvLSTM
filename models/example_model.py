from base.base_model import BaseModel
from models.conv_cell import ConvLSTMCell
from models.fc_attention import fc_attention_sum, fc_attention
from models.conv_attention import conv_attention_sum, conv_attention

import tensorflow as tf
from nets.mobilenet import mobilenet_v2


def print_list_to_file(list, file_name):
    thefile = open(file_name, 'w')
    for item in list:
        thefile.write("%s\n" % item)


class ExampleModel(BaseModel):

    def __init__(self, config):
        super(ExampleModel, self).__init__(config)

        # mobile_net nodes
        self.input_img = None
        #self.mn_layer_15 = None
        #self.mn_global_pool = None
        #self.mn_predictions = None
        self.mn_mobilenet_saver = None

        # complete network nodes
        self.fc_img = None
        self.conv_img = None
        self.conv_img_out = None
        self.ys = None
        self.Lr = None
        self.fc_loss = None
        self.conv_loss = None
        self.train_op = None

        self.saver = None
        self.mobilenet_saver = None
        self.lstm_saver = None

        self.fc_pred = None
        self.conv_pred = None
        self.loss = None
        self.prob = None
        self.alphas = None
        self.v = None
        # todo: is that the correct initialization?
        self.is_training = None

        self.build_model()
        self.init_global_saver()
        self.init_lstm_saver()

    def build_model(self):
        # training mode
        is_training = tf.placeholder(tf.bool)

        # y - supervision
        ys = tf.placeholder(tf.float32, [None, self.config.n_classes])

        # Learning rate
        Lr = tf.placeholder(tf.float32)

        # dropout probability
        prob = tf.placeholder_with_default(1.0, shape=())

        # input is already centered and cropped to mobile_net input size
        input_img = tf.placeholder(tf.float32, [None, self.config.n_steps] + self.config.frame_size)

        # reshape to match mobilenet input
        input_img_mn = tf.reshape(input_img, [-1] + self.config.frame_size)

        #  start by building the feature extractor:
        with tf.variable_scope("mobile_net"):
            # Define the model:
            # Note: arg_scope is optional for inference.
            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                last_layer_logits, end_points = mobilenet_v2.mobilenet(input_img_mn)

            conv_img = end_points[self.config.mobilenet_out_layer]
            fc_img = end_points['global_pool']
            predictions = end_points['Predictions']
            # initialize the mobilenet saver
            self.init_mobilenet_saver()

        # todo: get the shape from output!
        # reshape batch to have n_steps dimension
        conv_img_re = tf.stop_gradient(tf.reshape(conv_img, [-1, self.config.n_steps] + self.config.conv_input_shape + [self.config.channels]))
        fc_img_re = tf.reshape(fc_img, [-1, self.config.n_steps] + [1, 1, self.config.n_fc_inputs])
        fc_img_re = tf.stop_gradient(tf.squeeze(fc_img_re, [2, 3]))

        fc_img_out = self.FC_LSTM(fc_img_re, True)
        conv_img_out, alphas, v = self.CONV_LSTM(conv_img_re, True)

        fc_img_drop = tf.nn.dropout(fc_img_out, prob)
        conv_img_drop = tf.nn.dropout(conv_img_out, prob)
        conv_img_drop = tf.nn.max_pool(conv_img_drop, [1, self.config.conv_input_shape[0],
                                                       self.config.conv_input_shape[1], 1], [1, 1, 1, 1], padding='VALID')
        conv_img_drop = tf.reshape(conv_img_drop, [-1, self.config.n_filters])

        with tf.variable_scope("FC"):
            fc_result = self.FC_layer(fc_img_drop, prob)
            tf.get_variable_scope().reuse_variables()
            conv_result = self.FC_layer(conv_img_drop, prob)

        fc_pred = tf.nn.softmax(fc_result)
        conv_pred = tf.nn.softmax(conv_result)

        fc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_result, labels=ys))
        conv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conv_result, labels=ys))
        loss = fc_loss + conv_loss

        opt = tf.train.AdamOptimizer(Lr)

        # Retrieve all trainable variables defined in graph
        tvs = [v for v in tf.trainable_variables() if v.name[:10] != 'mobile_net']

        # Creation of a list of variables with the same shape as the trainable ones
        # initialized with 0s
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        # Calls the compute_gradients function of the optimizer to obtain... the list of gradients
        gvs = opt.compute_gradients(loss, tvs)

        # Adds to each element from the list you initialized earlier with zeros its gradient
        # (works because accum_vars and gvs are in the same order)
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]


        # Define the training step (part with variable value update)
        train_op = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)], global_step=self.global_step_tensor)
        # train_op = opt.apply_gradients([(accum_vars[i], tv) for i, tv in enumerate(tvs)], global_step=self.global_step_tensor)


        #train_op = tf.train.AdamOptimizer(Lr).minimize(loss, global_step=self.global_step_tensor)

        #saver = tf.train.Saver()

        # maintain the complete model nodes and points of interaction
        self.fc_img = fc_img
        self.conv_img = conv_img
        self.conv_img_out = conv_img_out
        self.ys = ys
        self.Lr = Lr
        self.fc_loss = fc_loss
        self.conv_loss = conv_loss
        self.train_op = train_op
        self.accum_ops = accum_ops
        self.accum_vars = accum_vars
        self.gvs = gvs
        self.zero_ops = zero_ops
        #self.saver = saver
        self.fc_pred = fc_pred
        self.conv_pred = conv_pred
        self.loss = loss
        self.prob = prob
        self.alphas = alphas
        self.v = v
        self.is_training = is_training

        self.input_img = input_img
        self.mn_predictions = predictions

    def FC_LSTM(self, X_spa, attention):
        weights_img = tf.Variable(tf.random_normal([self.config.n_fc_inputs, self.config.n_hidden_units]))
        biases_img = tf.Variable(tf.constant(0.1, shape=[self.config.n_hidden_units, ]))

        X_spa = tf.reshape(X_spa, [self.config.batch_size * self.config.n_steps, self.config.n_fc_inputs])
        X_spa = tf.matmul(X_spa, weights_img) + biases_img
        X_spa = tf.reshape(X_spa, [-1, self.config.n_steps, self.config.n_hidden_units])

        cell_spa = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden_units)
        mlstm_cell_spa = tf.contrib.rnn.MultiRNNCell([cell_spa for _ in range(self.config.n_layers)], state_is_tuple=True)
        init_state_spa = mlstm_cell_spa.zero_state(self.config.batch_size, dtype=tf.float32)
        outputs_spa, final_state_spa = tf.nn.dynamic_rnn(mlstm_cell_spa, X_spa, initial_state=init_state_spa,
                                                         time_major=False)

        attention_output_spa = fc_attention_sum(outputs_spa, self.config.fc_attention_size)
        if attention:
            attention_output_spa = tf.layers.batch_normalization(attention_output_spa)
            return attention_output_spa
        else:
            outputs_spa = tf.layers.batch_normalization(outputs_spa)
            outputs_spa = tf.reduce_sum(outputs_spa, axis=1)
            return outputs_spa

    def CONV_LSTM(self, conv_img, attention):
        img_cell = ConvLSTMCell(self.config.conv_input_shape, self.config.n_filters, self.config.kernel)
        img_outputs, img_state = tf.nn.dynamic_rnn(img_cell, conv_img, dtype=conv_img.dtype, time_major=True)
        if attention:
            img_attention_output, alphas, v = conv_attention_sum(img_outputs, self.config.attention_kernel)
            img_attention_output = tf.layers.batch_normalization(img_attention_output)
            return img_attention_output, alphas, v
        else:
            alphas = v = 0
            img_outputs = tf.layers.batch_normalization(img_outputs)
            img_outputs = tf.reduce_sum(img_outputs, axis=1)
            return img_outputs, alphas, v

    def FC_layer(self, inputs, drop_prob):
        weights2 = tf.get_variable("weights2", [self.config.n_hidden_units, self.config.n_classes],
                                   initializer=tf.truncated_normal_initializer())
        biases2 = tf.get_variable("biases2", [self.config.n_classes],
                                  initializer=tf.truncated_normal_initializer())
        result = tf.nn.dropout((tf.matmul(inputs, weights2) + biases2), drop_prob)
        return result

    # the saver node was created already - this actually restores the variables
    def restore_mobile_net(self, sess):
        if self.config.mobilenet_base == 'imagenet':
            print("Loading mobilenet model from checkpoint {} ...\n".format(self.config.mobile_net_ckpt))
            self.mobilenet_saver.restore(sess, self.config.mobile_net_ckpt)
        else:
            model_name = self.get_latest_model_name()
            print("Loading mobilenet model from checkpoint {} ...\n".format(model_name))
            self.mobilenet_saver.restore(sess, model_name)


    # just creates the saver node
    def init_mobilenet_saver(self):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobile_net')

        if self.config.mobilenet_base == 'imagenet':
            var_list = {var.op.name[11:]: var for var in var_list}
        # restoring from my checkpoint

        self.mobilenet_saver = tf.train.Saver(var_list, max_to_keep=self.config.max_to_keep)

    # just creates the saver node
    def init_lstm_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # this saver deals with all cariables except mobile net.
        restore_var = [v for v in tf.all_variables() if v.name[:10] != 'mobile_net']
        self.lstm_saver = tf.train.Saver(restore_var, max_to_keep=self.config.max_to_keep)

    # just creates the saver node
    def init_global_saver(self):

        var_names = [v.name for v in tf.all_variables()]
        with open('all_vars.txt', 'w') as f:
            for item in var_names:
                f.write("%s\n" % item)

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def compute_alphas_attention(full_model_specs, sess, batch_fc_img, batch_conv_img, batch_labels):
        alphas = full_model_specs['alphas']
        v = full_model_specs['v']

        al, v = sess.run([alphas, v], feed_dict={
            full_model_specs['fc_img']: batch_fc_img,
            full_model_specs['conv_img']: batch_conv_img,
            full_model_specs['ys']: batch_labels,
            full_model_specs['prob']: 1.0
        })

        return al, v
















