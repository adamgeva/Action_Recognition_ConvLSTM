from base.base_test import BaseTest
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc
import cv2
from train_faster import predict
from utils.video_fifo import VideoFIFO
import utils.utils_data as utils_data

def accuracy(a, b):
    c = np.equal(a, b).astype(float)
    acc = sum(c) / len(c)
    return acc


class ExampleTesterOneSeq(BaseTest):
    def __init__(self, sess, model, pred, clip_path, config):
        # Assign all class attributes
        self.model = model
        self.faster_pred = pred
        self.config = config
        self.sess = sess

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.clip_path = clip_path

        self.label_dict = utils_data.read_classes(config.classInd)
        self.label_dict_inv = {v: k for k, v in self.label_dict.items()}

    def test(self):

        capture = cv2.VideoCapture(self.clip_path)


        fps = capture.get(cv2.CAP_PROP_FPS)

        frame_gap = int(round(fps / self.config.target_fps))
        frame_num = 0

        frame_height = int(capture.get(3))
        frame_width = int(capture.get(4))

        curr_fifo = VideoFIFO(self.config, self.config.n_timesteps, frame_width, frame_height)

        flag = 1
        # read video frames
        while flag is not 0:
            flag, frame = capture.read()

            if frame_num % frame_gap == 0:
                curr_fifo.add_frame(frame)

                # run faster to detect region every frame
                res = predict(self.faster_pred, frame)

                # get segments block of previous 13 frames - bb interpolation
                human_bb = self.filter_bb(res)
                # skip if no detection
                if human_bb != [0,0,0,0]:
                    for human_detection in human_bb:
                        curr_block = curr_fifo.recover_block(human_detection, self.config.new_frame_size[0],
                                                             self.config.new_frame_size[1])

                        # run model to get predictions
                        predictions_add, predictions_mul = self.test_step(curr_block)

                        # draw predictions
                        print(self.label_dict_inv[int(predictions_add)])
                        print(self.label_dict_inv[int(predictions_mul)])
                        cv2.rectangle(frame, (human_detection[0], human_detection[1]), (human_detection[2], human_detection[3]),
                                      color=(0, 0, 255), thickness=1)

            # plot frame
            cv2.imshow('vid', frame)
            cv2.waitKey(100)
            #cv2.destroyAllWindows()
            frame_num += 1

    def test_step(self, block):

        prob = 1.0
        batch_frames = block
        batch_frames = np.expand_dims(batch_frames, axis=0)
        feed_dict = {
            self.model.is_training: False,
            self.model.input_img: batch_frames,
            self.model.prob: prob
        }

        fc_score, conv_score = self.sess.run([self.model.fc_pred, self.model.conv_pred], feed_dict)

        # calc accuracy of the batch
        fc_score = np.reshape(np.array(fc_score), (self.config.batch_size, self.config.n_classes))  # (batch_size, n_classes)
        conv_score = np.reshape(np.array(conv_score), (self.config.batch_size, self.config.n_classes))

        # fusion by addition
        fus_add = np.add(fc_score, conv_score)
        predictions_add = np.argmax(fus_add, axis=1)

        # fusion by multiplication
        fus_mul = np.multiply(fc_score, conv_score)
        predictions_mul = np.argmax(fus_mul, axis=1)

        return predictions_add, predictions_mul

    @staticmethod
    def filter_bb(res):
        bb_man = []
        no_man_detection = True

        for detection in res:
            bb = detection[0]
            score = detection[1]
            id = detection[2]
            if (id == 1) and score > 0.8:  # man detected
                bb_man.append(bb)
                no_man_detection = False

        if no_man_detection:
            print('no man detection!!!')
            bb_man = [0, 0, 0, 0]

        return bb_man
