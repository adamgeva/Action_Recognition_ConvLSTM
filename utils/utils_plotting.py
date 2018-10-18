import cv2
import numpy as np

def plot_attention(atten, name):
    norm_atten = normalize_im(atten)
    for i in range(13):
        curr_im = norm_atten[0, i, :, :, 0]
        curr_im = cv2.resize(curr_im, (224, 224), interpolation=cv2.INTER_LINEAR)
        #cv2.imshow(name, curr_im)
        #cv2.waitKey(0)
        cv2.imwrite(name + str(i) + '.jpg', curr_im)


def plot_input_images(input_images, name):
    input_images = normalize_im(input_images)
    for i in range(13):
        curr_im = input_images[0, i, :, :, :]
        cv2.imshow(name, curr_im)
        cv2.waitKey(0)
        #cv2.imwrite(name + str(i) + '.jpg', curr_im)


def normalize_im(im):
    im_new = im.astype(np.float64)

    minimum = im_new.min()

    maximum = im_new.max()

    im_norm = 255 * (im_new - minimum) / (maximum - minimum)

    im_norm = im_norm.astype(np.uint8)

    return im_norm


def plot_h_abs(h):
    h_new = np.abs(h)
    h_new = np.sum(h_new, 4)
    h_new = np.expand_dims(h_new, axis=4)
    plot_attention(h_new, 'y')