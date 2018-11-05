import cv2
import numpy as np


def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def crop_center_image(image, rec_len):
    # crop center
    (h, w) = image.shape[:2]
    x_1 = (w - rec_len) // 2
    x_2 = (h - rec_len) // 2
    cropped = image[x_2:x_2+rec_len, x_1:x_1+rec_len]
    return cropped


# preprocessing:

video_path = '/hdd/UCF-ARG/rooftop_clips_stab_crop/running/person01_03_rooftop_running0.avi'
#video_path = '/home/ADAMGE/Downloads/test_jog.mp4'
target_fps = 25.0
n_steps = 100
resize_height = 256
crop_size = 224


clip_frames = []
clip_frames_flow = []
bit = 0
capture = cv2.VideoCapture(video_path)
fps = capture.get(cv2.CAP_PROP_FPS)

frame_gap = int(round(fps / target_fps))
frame_num = 1

ret, frame1 = capture.read()
resized_frame1 = image_resize(frame1, height = resize_height)
prvs = cv2.cvtColor(resized_frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# extract features
while (capture.isOpened()) & (int(round(frame_num / frame_gap)) < n_steps) & (bit == 0):
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
        # RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = image_resize(image, height = resize_height)
        res = resized_frame.astype('float32') / 128. - 1
        res = crop_center_image(res, crop_size)
        clip_frames.append(res)

        # plotting:
        res_to_plot = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        res_to_plot = res_to_plot + 1.0 / 2.0
        cv2.imshow("Vid", res_to_plot)
        key_pressed = cv2.waitKey(10)  # Escape to exit

        # FLOW
        image_flow = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(prvs, image_flow, None)
        flow[flow > 20] = 20
        flow[flow < -20] = -20
        flow = flow / 20.
        flow = crop_center_image(flow, crop_size)
        clip_frames_flow.append(flow)

        # potting:
        flow_temp = (flow + 1.0) / 2.0
        last_channel = np.zeros((crop_size,crop_size), dtype=float) + 0.5
        flow_to_plot = np.dstack((flow_temp, last_channel))
        cv2.imshow("Vid-flow", flow_to_plot)
        key_pressed = cv2.waitKey(10)  # Escape to exit


        #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #hsv[..., 0] = ang * 180 / np.pi / 2
        #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #cv2.imshow('frame2', bgr)
        #k = cv2.waitKey(30) & 0xff

        prvs = image_flow

    frame_num += 1

capture.release()

frames = np.array(clip_frames)
frames_flow = np.array(clip_frames_flow)
frames = np.expand_dims(frames, axis=0)
frames_flow = np.expand_dims(frames_flow, axis=0)

np.save('rgb.npy', frames)
np.save('flow.npy', frames_flow)



