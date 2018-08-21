import numpy as np
import utils.utils_data as utils_data
import utils.utils_video as utils_video
import cv2
import time


class DataGenerator:
    def __init__(self, model, config, sess, mode):
        self.config = config

        # data_generator receives the model and session because it runs the first part of the model -
        # mobile_net to return batch features
        self.model = model
        self.sess = sess
        # read classes dict
        self.label_dict = utils_data.read_classes(config.classInd)

        # load data here (shuffle inside)
        if mode == "train":
            self.lines, self.labels, self.len_lines = utils_data.read_data(config.train_list, self.label_dict, True)
        elif mode == "test":
            self.lines, self.labels, self.len_lines = utils_data.read_data(config.test_list, self.label_dict, False)

        # feeder state
        self._curr_line_num = 0

    # resets the feeder to the first example
    def reset_feeder(self):
        self._curr_line_num = 0

    # returns the batch fc and conv features and labels
    def next_batch(self):
        time1 = time.time()

        batch_fc_img = np.zeros((self.config.batch_size,
                                 self.config.n_steps,
                                 self.config.n_fc_inputs), dtype=np.float32)  # (60, 40, 4096)

        batch_conv_img = np.zeros((self.config.batch_size,
                                   self.config.n_steps,
                                   self.config.conv_input_shape[0],
                                   self.config.conv_input_shape[1],
                                   self.config.channels), dtype=np.float32)  # (60, 7, 7, 512)

        batch_labels = np.zeros((self.config.batch_size, self.config.n_classes), dtype=np.int8)

        for example_ind in range(self.config.batch_size):
            if self.end_of_data():
                # if we finished the data, the rest of the batch array is zeros.
                break
            else:
                # get next path and class
                curr_video_full_path, curr_video_class = self.get_next_example()

                # create the label example
                one_hot = np.zeros(self.config.n_classes, dtype=np.int8)
                one_hot[self.label_dict[curr_video_class.split("_")[0]]] = 1
                label = np.expand_dims(one_hot, axis=0)

                # extract features of example
                bit_correct, fc_img_fea, conv_img_fea = get_fc_conv_features(self.model, self.config,
                                                                             self.sess, curr_video_full_path)

                if bit_correct == 1:
                    print('ERROR: skipping clip...')
                    continue

                # assign to the big array
                batch_fc_img[example_ind] = fc_img_fea
                batch_conv_img[example_ind] = conv_img_fea
                batch_labels[example_ind] = label

        time2 = time.time()

        print("batch_read_time:", '{0:.2f}'.format(time2 - time1), "s")

        return batch_fc_img, batch_conv_img, batch_labels

    # did we reach the end of the line list?
    def end_of_data(self):
        return self._curr_line_num == self.len_lines

    def get_next_example(self):
        # returns the video full path, class, first frame to read from
        line = self.lines[self._curr_line_num]
        curr_video_full_path, curr_video_class = utils_video.line_to_path(line, self.config.UCF11_path)
        self.update_state()
        return curr_video_full_path, curr_video_class

    def update_state(self):
        self._curr_line_num += 1


# use CNN to extract features - these features are then used as input for the LSTM networks
def get_fc_conv_features(model, config, sess, video_path):

    num_conv_features = (config.conv_input_shape[0], config.conv_input_shape[1], config.channels)

    bit = 0
    capture = cv2.VideoCapture(video_path)

    # array to hold all frame fc features
    fc_features = np.zeros((config.n_steps, config.n_fc_inputs))

    # array to hold all frame conv features
    conv_features = np.zeros((config.n_steps,) + num_conv_features)

    frame_num = 0
    # extract features
    while (capture.isOpened()) & (frame_num < config.n_steps) & (bit == 0):
        flag, frame = capture.read()
        if flag == 0:
            bit = 1
            print("******ERROR: Could not read frame in " + video_path + " frame_num: " + str(frame_num))
            break

        #name = params['res_vids_path'] + str(frame_num) + 'frame.jpg'
        #cv2.imwrite(name, frame)
        #cv2.imshow("Video", frame)
        #key_pressed = cv2.waitKey(10)  # Escape to exit

        # process frame
        centered_image = utils_video.val_reprocess(config, frame)

        # forward vgg (including vgg pre-processing)
        #res2, res = self._my_sess.run([self._pool5_features,
        #                      self._fc6_features], {self._input_img: centered_image})

        res2, res = sess.run([model.layer_15, model.global_pool],
                             {model.input_img: centered_image})

        #pred = sess.run(model_specs['predictions'], {model_specs['input_img']: centered_image})
        #label_map = imagenet.create_readable_names_for_imagenet_labels()
        #print("Top 1 prediction: ", pred.argmax(), label_map[pred.argmax()], pred.max())

        # collect to all video features
        fc_features[frame_num, :] = res[0, :]
        #np.append(fc_features, res[0, :], axis=0)
        conv_features[frame_num, :, :, :] = res2[0, :, :, :]
        #np.append(conv_features, res2[0, :, :, :], axis=0)

        #print(np.shape(res))
        #print(np.shape(res2))
        #input_img = sess.run(vgg_extractor.return_input(), {image: img1})

        frame_num += 1
        #if key_pressed == 27:
        #    break

    capture.release()

    return bit, fc_features, conv_features
    # soft max on output
    #res1 = np.exp(res)
    #res2 = res1 / np.sum(res1)
    #indices = np.argsort(res2)
    # print the top 10 predictions
    #print(indices[0][-10:])