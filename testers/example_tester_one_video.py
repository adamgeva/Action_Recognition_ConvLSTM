from base.base_test import BaseTest
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc
import cv2
from train_faster import predict, main_run_faster

from utils.video_fifo import VideoFIFO
from utils.utils_video import filter_bb
import utils.utils_data as utils_data

import copy


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

        # object tracker
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        initBB = False
        # grab the appropriate object tracker using our dictionary of


        # video capture
        capture = cv2.VideoCapture(self.clip_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        # used to deal with frame rate corrections
        frame_gap = int(round(fps / self.config.target_fps))
        frame_num = 0
        frame_height = int(capture.get(4))
        frame_width = int(capture.get(3))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('IR_example.avi', fourcc, fps, (frame_width, frame_height))

        # initialize the frame fifo
        curr_fifo = VideoFIFO(self.config, self.config.n_timesteps, frame_width, frame_height)

        # initialize OpenCV's special multi-object tracker
        trackers = cv2.MultiTracker_create()

        detections = []
        predictions = []

        flag = 1
        # read video frames
        while True:
            flag, frame = capture.read()

            # end of movie
            if flag==0:
                break

            # use FASTER for detection as long as the tracker is not initialized
            if not initBB:
                # run faster to detect region every frame
                res = predict(self.faster_pred, frame)

                # get segments block of previous 13 frames - bb interpolation
                human_bb = filter_bb(res)

                human_bb = [0,0,0,0]
                # skip if no detection
                if np.all([x == 0 for x in human_bb]):
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                    human_bb = cv2.selectROI("Frame", frame, fromCenter=False,
                                           showCrosshair=True)

                    # OpenCV object tracker objects
                    tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

                    # start OpenCV object tracker using the supplied bounding box

                    trackers.add(tracker, frame, human_bb)
                    human_detection_det = (human_bb[0], human_bb[1], human_bb[2] + human_bb[0], human_bb[3] + human_bb[1])
                    detections.append(copy.copy(human_detection_det))
                    predictions.append(0)

                else:
                    for det in human_bb:
                        # OpenCV object tracker objects
                        tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

                        # start OpenCV object tracker using the supplied bounding box
                        # coordinates, then start the FPS throughput estimator as well
                        human_detection_tup = (det[0], det[1], det[2]-det[0], det[3]-det[1])

                        trackers.add(tracker, frame, human_detection_tup)

                        detections.append(copy.copy(det))
                        predictions.append(0)

                initBB = True

            else:

                (success, human_detection_tups) = trackers.update(frame)
                for i, human_detection_tuple in enumerate(human_detection_tups):
                    # grab the new bounding box coordinates of the object
                    (x, y, w, h) = [int(v) for v in human_detection_tuple]
                    human_detection = [x, y, x + w, y + h]
                    detections[i] = human_detection

            if frame_num % frame_gap == 0:
                # add to fifo according to the correct frame rate
                curr_fifo.add_frame(frame)

                for i, human_det in enumerate(detections):
                    curr_block = curr_fifo.recover_block(human_det, self.config.new_frame_size[0],
                                                         self.config.new_frame_size[1], frame_num)

                    # run model to get predictions
                    predictions_add, predictions_mul = self.test_step(curr_block)
                    predictions[i] = predictions_mul
                    # draw predictions
                    print(self.label_dict_inv[int(predictions_add)])
                    print(self.label_dict_inv[int(predictions_mul)])

            frame_rec = frame.copy()

            for i, h in enumerate(detections):
                cv2.rectangle(frame_rec, (h[0], h[1]), (h[2], h[3]),
                              color=(0, 0, 255), thickness=1)

                cv2.putText(frame_rec, self.label_dict_inv[int(predictions[i])],
                            (h[0], h[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1)

            out.write(frame_rec)

            # plot frame
            cv2.imshow('vid', frame_rec)
            cv2.waitKey(100)
            #cv2.destroyAllWindows()
            frame_num += 1

        # When everything done, release the capture
        capture.release()
        out.release()
        cv2.destroyAllWindows()

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
