from base.base_test import BaseTest
from tqdm import tqdm
import numpy as np
import cv2
import gc


def accuracy(a, b):
    c = np.equal(a, b).astype(float)
    acc = sum(c) / len(c)
    return acc

def print_alphas_conv(alp):
    for i in range(np.shape(alp)[1]):
        curr_im = alp[:,i,:,:,:]

def normalize_im(im):
    im_new = im.astype(np.float64)

    minimum = im_new.min()

    maximum = im_new.max()

    im_norm = 255 * (im_new - minimum) / (maximum - minimum)

    im_norm = im_norm.astype(np.uint8)

    return im_norm


class ExampleTesterPlotAttention(BaseTest):
    def __init__(self, sess, model, data_test, config, logger):
        super(ExampleTesterPlotAttention, self).__init__(sess, model, config, logger)

        self.data_test = data_test

        # calculate number of training and validation steps per epochs
        self.num_iter_data = data_test.len_lines // self.config.batch_size

    def test(self):

        losses_val = []
        accs_add_val = []
        accs_mul_val = []
        predictions_add_val = []
        predictions_mul_val = []
        gt_classes_val = []

        loop_test = tqdm(range(self.num_iter_data))

        # iterate over steps (batches)
        for _ in loop_test:
            accu_add, accu_mul, loss, predictions_add, predictions_mul, gt_classes, a = self.test_step()
            losses_val.append(loss)
            accs_add_val.append(accu_add)
            accs_mul_val.append(accu_mul)

            # collect also the actual predictions to create confusion matrix
            predictions_add_val = np.append(predictions_add_val, predictions_add)
            predictions_mul_val = np.append(predictions_mul_val, predictions_mul)
            gt_classes_val = np.append(gt_classes_val, gt_classes)

        loss_val_epoch = np.mean(losses_val)
        accs_add_val_epoch = np.mean(accs_add_val)
        accs_mul_val_epoch = np.mean(accs_mul_val)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss_validation': loss_val_epoch,
            'accuracy_add_validation': accs_add_val_epoch,
            'accuracy_multiply_validation': accs_mul_val_epoch
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        labels = sorted(self.data_test.label_dict, key=self.data_test.label_dict.get)
        self.logger.confusion_mat(cur_it, labels, [self.data_test.label_dict_inv[int(i)] for i in gt_classes_val],
                                  [self.data_test.label_dict_inv[int(i)] for i in predictions_add_val],
                                  [self.data_test.label_dict_inv[int(i)] for i in predictions_mul_val], 'test')

        print_alphas_conv(a)

    def test_step(self):

        prob = 1.0
        batch_frames, batch_labels = self.data_test.next_batch()

        feed_dict = {
            self.model.is_training: False,
            self.model.input_img: batch_frames,
            self.model.ys: batch_labels,
            self.model.prob: prob
        }

        fc_score, conv_score, loss, a, a_fc, im_outputs = self.sess.run([self.model.fc_pred, self.model.conv_pred,
                                                    self.model.loss, self.model.alphas,
                                                    self.model.alphas_fc, self.model.im_outputs], feed_dict)

        # calc accuracy of the batch
        fc_score = np.reshape(np.array(fc_score), (self.config.batch_size, self.config.n_classes))  # (batch_size, n_classes)
        conv_score = np.reshape(np.array(conv_score), (self.config.batch_size, self.config.n_classes))

        gt_classes = np.nonzero(batch_labels)[1]

        # fusion by addition
        fus_add = np.add(fc_score, conv_score)
        predictions_add = np.argmax(fus_add, axis=1)
        accu_add = accuracy(predictions_add, gt_classes)

        # fusion by multiplication
        fus_mul = np.multiply(fc_score, conv_score)
        predictions_mul = np.argmax(fus_mul, axis=1)
        accu_mul = accuracy(predictions_mul, gt_classes)

        return accu_add, accu_mul, loss, predictions_add, predictions_mul, gt_classes, a

