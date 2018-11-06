import numpy as np
import utils.utils_data as utils_data
import utils.utils_video as utils_video
import cv2
import time


class DataGenerator:
    def __init__(self, model, config, sess, data_name, shuffle, augment):
        self.config = config

        # data_generator receives the model and session because it runs the first part of the model -
        # mobile_net to return batch features
        self.model = model
        self.sess = sess
        # read classes dict
        self.label_dict = utils_data.read_classes(config.classInd)
        self.label_dict_inv = {v: k for k, v in self.label_dict.items()}

        self.augment = augment

        # load data here (shuffle inside)
        if data_name == 'train':
            data_list = config.train_list
        elif data_name == 'validate':
            data_list = config.val_list
        else:
            data_list = config.test_list

        self.lines, self.labels, self.len_lines = utils_data.read_data(data_list, self.label_dict, shuffle, config.data)

        # feeder state
        self._curr_line_num = 0

    # resets the feeder to the first example
    def reset_feeder(self):
        self._curr_line_num = 0

    # returns the batch fc and conv features and labels
    def next_batch(self):
        time1 = time.time()
        eod = False
        batch_frames = np.zeros((self.config.batch_size,
                                 self.config.n_steps,
                                 self.config.frame_size[0],
                                 self.config.frame_size[1],
                                 self.config.frame_size[2],), dtype=np.float32)  # (8, 40, 224, 224, 3)

        batch_labels = np.zeros((self.config.batch_size, self.config.n_classes), dtype=np.int8)

        example_ind = 0
        while example_ind < self.config.batch_size:
            if self.end_of_data():
                # if we finished the data, the rest of the batch array is zeros.
                eod = True
                break
            else:
                # get next path and class
                curr_video_full_path, curr_video_class = self.get_next_example()

                # create the label example
                one_hot = np.zeros(self.config.n_classes, dtype=np.int8)
                one_hot[self.label_dict[curr_video_class]] = 1
                label = np.expand_dims(one_hot, axis=0)

                # extract frames
                bit_correct, frames = get_clip_frames(self.config, curr_video_full_path)

                if bit_correct == 1:
                    print('ERROR: skipping clip...')
                    continue

                # augment example if required
                if self.augment:
                    frames = utils_video.augment_frames(frames)

                # assign to the big array
                batch_frames[example_ind] = frames
                batch_labels[example_ind] = label
                example_ind += 1

        time2 = time.time()

        print("batch_read_time:", '{0:.2f}'.format(time2 - time1), "s")

        return batch_frames, batch_labels, eod

    # did we reach the end of the line list?
    def end_of_data(self):
        return self._curr_line_num == self.len_lines

    def get_next_example(self):
        # returns the video full path, class, first frame to read from
        line = self.lines[self._curr_line_num]
        if self.config.data == "UCF":
            curr_video_full_path, curr_video_class = utils_video.line_to_path(line, self.config.UCF_ARG_path)
        elif self.config.data == "SDHA":
            curr_video_full_path, curr_video_class = utils_video.line_to_path_SDHA(line, self.config.SDHA_2010_path)
        elif self.config.data == "Combined":
            curr_video_full_path, curr_video_class = utils_video.line_to_path_Combined(line, self.config.Combined_path)
        elif self.config.data == "IR":
            curr_video_full_path, curr_video_class = utils_video.line_to_path_IR(line, self.config.IR_path)
        elif self.config.data == "HMDB":
            curr_video_full_path, curr_video_class = utils_video.line_to_path_HMDB(line, self.config.HMDB_path, self.label_dict_inv)
        self.update_state()
        return curr_video_full_path, curr_video_class

    def update_state(self):
        self._curr_line_num += 1


# # use CNN to extract features - these features are then used as input for the LSTM networks
# def get_fc_conv_features(model, config, sess, video_path, is_training):
#
#     num_conv_features = (config.conv_input_shape[0], config.conv_input_shape[1], config.channels)
#
#     bit = 0
#     capture = cv2.VideoCapture(video_path)
#
#     # array to hold all frame fc features
#     fc_features = np.zeros((config.n_steps, config.n_fc_inputs))
#
#     # array to hold all frame conv features
#     conv_features = np.zeros((config.n_steps,) + num_conv_features)
#
#     frame_num = 0
#     # extract features
#     while (capture.isOpened()) & (frame_num < config.n_steps) & (bit == 0):
#         flag, frame = capture.read()
#         if flag == 0:
#             bit = 1
#             print("******ERROR: Could not read frame in " + video_path + " frame_num: " + str(frame_num))
#             break
#
#         #name = params['res_vids_path'] + str(frame_num) + 'frame.jpg'
#         #cv2.imwrite(name, frame)
#         #cv2.imshow("Vid", frame)
#         #key_pressed = cv2.waitKey(10)  # Escape to exit
#
#         # process frame
#         centered_image = utils_video.val_reprocess(config, frame)
#
#         # forward vgg (including vgg pre-processing)
#         #res2, res = self._my_sess.run([self._pool5_features,
#         #                      self._fc6_features], {self._input_img: centered_image})
#
#         res2, res = sess.run([model.mn_layer_15, model.mn_global_pool],
#                              {model.mn_input_img: centered_image, model.is_training: is_training})
#
#         #pred = sess.run(model_specs['predictions'], {model_specs['input_img']: centered_image})
#         #label_map = imagenet.create_readable_names_for_imagenet_labels()
#         #print("Top 1 prediction: ", pred.argmax(), label_map[pred.argmax()], pred.max())
#
#         # collect to all video features
#         fc_features[frame_num, :] = res[0, :]
#         #np.append(fc_features, res[0, :], axis=0)
#         conv_features[frame_num, :, :, :] = res2[0, :, :, :]
#         #np.append(conv_features, res2[0, :, :, :], axis=0)
#
#         #print(np.shape(res))
#         #print(np.shape(res2))
#         #input_img = sess.run(vgg_extractor.return_input(), {image: img1})
#
#         frame_num += 1
#         #if key_pressed == 27:
#         #    break
#
#     capture.release()
#
#     return bit, fc_features, conv_features
#     # soft max on output
#     #res1 = np.exp(res)
#     #res2 = res1 / np.sum(res1)
#     #indices = np.argsort(res2)
#     # print the top 10 predictions
#     #print(indices[0][-10:])


def get_clip_frames(config, video_path):

    clip_frames = []
    bit = 0
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_gap = int(round(fps / config.target_fps))

    if config.random_start:
        start_frame = int(np.random.rand() * (length - int(config.n_steps * frame_gap) - 1))

        # skip first frames
        for j in range(start_frame):
            flag, frame = capture.read()

    frame_num = 0
    # extract features
    while (capture.isOpened()) & (int(round(frame_num / frame_gap)) < config.n_steps) & (bit == 0):
        flag, frame = capture.read()
        if flag == 0:
            bit = 1
            print("******ERROR: Could not read frame in " + video_path + " frame_num: " + str(frame_num))
            break

        #name = params['res_vids_path'] + str(frame_num) + 'frame.jpg'
        #cv2.imwrite(name, frame)
        #cv2.imshow("Vid", frame)
        #key_pressed = cv2.waitKey(10)  # Escape to exit

        # process frame (according to the correct frame rate)
        if frame_num % frame_gap == 0:
            centered_image = utils_video.val_reprocess(config, frame)
            clip_frames.append(centered_image)
            #cv2.imshow("Vid", frame)
            #key_pressed = cv2.waitKey(30)  # Escape to exit

        frame_num += 1

    capture.release()

    frames = np.array(clip_frames)

    return bit, frames
