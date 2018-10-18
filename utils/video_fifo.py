import numpy as np
import utils.utils_video as utils_video
import cv2
import math

class VideoFIFO:
    def __init__(self, config, time_depth, width, height):
        zero_frame = np.zeros((height, width, 3), np.uint8)
        self.fifo = [zero_frame for i in range(time_depth)]
        self.config = config
        self.width = width
        self.height = height
        self.time_depth = time_depth

    # add a new frame and remove the last one
    def add_frame(self, frame):
        self.fifo.pop(0)
        self.fifo.append(frame)

    # recover block : gets bb of human and returns the cropped last block containing the human
    def recover_block(self, human_bb, new_width, new_height, frame_num):
        center_point = [0, 0]

        center_point[0] = (human_bb[1] + human_bb[3]) // 2
        center_point[1] = (human_bb[0] + human_bb[2]) // 2

        human_height = human_bb[3] - human_bb[1]

        #if (center_point[0]-(new_height // 2)) < 0:
        #    center_point[0] = (new_height // 2)
        #elif (center_point[0]+(new_height // 2)) > self.height:
        #    center_point[0] = self.height - (new_height // 2)

        #if (center_point[1] - (new_width // 2)) < 0:
        #    center_point[1] = (new_width // 2)
        #elif (center_point[1] + (new_width // 2)) > self.width:
        #    center_point[1] = self.width - (new_width // 2)

        block = np.zeros([self.time_depth, self.config.frame_size[0], self.config.frame_size[1], 3])

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(str(frame_num) + 'test.avi', fourcc, 10,
        #                      (new_width, new_height))

        for i in range(self.time_depth):
            #cv2.imshow('Window', self.fifo[12])
            #cv2.waitKey(0)
            #new_frame = self.fifo[i][int(center_point[0] - new_height // 2):int(center_point[0] + new_height // 2),
            #            int(center_point[1] - new_width // 2):int(center_point[1] + new_width // 2)]

            new_frame = self.get_new_res_frame(center_point, human_height, self.fifo[i], new_height, new_width)
            # adjust image size
        #    out.write(new_frame)
            #cv2.imshow('Window2', new_frame)
            #cv2.waitKey(0)

            centered_image = utils_video.val_reprocess(self.config, new_frame)
            block[i] = centered_image

        #out.release()
        return block

    def get_new_res_frame(self, center, human_h, frame, new_frame_h , new_frame_w):
        rate = human_h / self.config.avg_human_h
        # we are modifying according to the target height and keep the aspect ration of the frame
        h1 = center[0]
        h2 = self.height - center[0]
        w1 = center[1]
        w2 = self.width - center[1]
        t_h1 = t_h2 = (new_frame_h // 2) * rate
        top, bottom, left, right = 0, 0, 0, 0

        if h2 < t_h2:
            bottom = int(math.ceil(t_h2 - h2))
        if h1 < t_h1:
            top = int(math.ceil(t_h1 - h1))
        if w2 < t_h2:
            right = int(math.ceil(t_h2 - w2))
        if w1 < t_h1:
            left = int(math.ceil(t_h1 - w1))

        dst = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REFLECT, value=[0,0,0])

        new_center = [0, 0]
        new_center[0] = center[0] + top
        new_center[1] = center[1] + left
        new_frame_h_crop = int(new_frame_h * rate)
        new_frame_w_crop = int(new_frame_w * rate)

        new_dst = dst[int(new_center[0] - new_frame_h_crop // 2): int(new_center[0] + new_frame_h_crop // 2),
                  int(new_center[1] - new_frame_w_crop // 2): int(new_center[1] + new_frame_w_crop // 2)]

        final_dst = cv2.resize(new_dst, (new_frame_w, new_frame_h), interpolation=cv2.INTER_LINEAR)

        return final_dst





