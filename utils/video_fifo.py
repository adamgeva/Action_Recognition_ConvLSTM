import numpy as np
import utils.utils_video as utils_video
import cv2

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

        if (center_point[0]-(new_height // 2)) < 0:
            center_point[0] = (new_height // 2)
        elif (center_point[0]+(new_height // 2)) > self.height:
            center_point[0] = self.height - (new_height // 2)

        if (center_point[1] - (new_width // 2)) < 0:
            center_point[1] = (new_width // 2)
        elif (center_point[1] + (new_width // 2)) > self.width:
            center_point[1] = self.width - (new_width // 2)

        block = np.zeros([self.time_depth, self.config.frame_size[0], self.config.frame_size[1], 3])

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(frame_num) + 'test.avi', fourcc, 10,
                              (new_width, new_height))

        for i in range(self.time_depth):
            #cv2.imshow('Window', self.fifo[12])
            #cv2.waitKey(0)
            new_frame = self.fifo[i][int(center_point[0] - new_height // 2):int(center_point[0] + new_height // 2),
                        int(center_point[1] - new_width // 2):int(center_point[1] + new_width // 2)]
            # adjust image size
            out.write(new_frame)
            #cv2.imshow('Window2', new_frame)
            #cv2.waitKey(0)

            centered_image = utils_video.val_reprocess(self.config, new_frame)
            block[i] = centered_image

        out.release()
        return block

