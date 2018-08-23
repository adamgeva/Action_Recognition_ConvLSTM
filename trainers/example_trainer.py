from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import gc


def accuracy(a, b):
    c = np.equal(a, b).astype(float)
    acc = sum(c) / len(c)
    return acc


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data_train, data_validate, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, config, logger)

        self.data_train = data_train
        self.data_validate = data_validate

        # restore mobile net
        self.model.restore_mobile_net(self.sess)

        # calculate number of training and validation steps per epochs
        self.num_train_iter_per_epoch = data_train.len_lines // self.config.batch_size
        self.num_validation_iter_per_epoch = data_validate.len_lines // self.config.batch_size

    def train_epoch(self, curr_epoch):
        lr = self.config.basic_lr

        loop_train = tqdm(range(self.num_train_iter_per_epoch))

        # iterate over steps (batches)
        for _ in loop_train:
            accu_add, accu_mul, loss, _, _, _ = self.train_validate_step(lr, True)

            cur_it = self.model.global_step_tensor.eval(self.sess)
            summaries_dict = {
                'loss': loss,
                'accuracy_add': accu_add,
                'accuracy_multiply': accu_mul
            }
            self.logger.summarize(cur_it, summaries_dict=summaries_dict)

            # todo: is this still necessary?
            #gc.collect()

        # validate every few epochs
        if curr_epoch % self.config.val_interval == 0:
            losses_val = []
            accs_add_val = []
            accs_mul_val = []
            predictions_add_val = []
            predictions_mul_val = []
            gt_classes_val = []

            loop_validate = tqdm(range(self.num_validation_iter_per_epoch))

            # iterate over steps (batches)
            for _ in loop_validate:
                accu_add, accu_mul, loss, predictions_add, predictions_mul, gt_classes = self.train_validate_step(lr, False)
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

            labels = sorted(self.data_train.label_dict, key=self.data_train.label_dict.get)
            self.logger.confusion_mat(cur_it, labels, [self.data_train.label_dict_inv[int(i)] for i in gt_classes_val],
                                      [self.data_train.label_dict_inv[int(i)] for i in predictions_add_val],
                                      [self.data_train.label_dict_inv[int(i)] for i in predictions_mul_val])

        # save model
        self.model.save(self.sess)

        # reset feeders for the next epoch
        self.data_train.reset_feeder()
        self.data_validate.reset_feeder()

    def train_validate_step(self, lr, is_training):

        # get next batch
        if is_training:
            prob = 0.5
            batch_fc_img, batch_conv_img, batch_labels = self.data_train.next_batch()
        else:
            prob = 1.0
            batch_fc_img, batch_conv_img, batch_labels = self.data_validate.next_batch()

        feed_dict = {
            self.model.fc_img: batch_fc_img,
            self.model.conv_img: batch_conv_img,
            self.model.ys: batch_labels,
            self.model.Lr: lr,
            self.model.is_training: is_training,
            self.model.prob: prob
        }

        if is_training:
            _, fc_score, conv_score, loss = self.sess.run([self.model.train_op, self.model.fc_pred,
                                                                 self.model.conv_pred, self.model.loss], feed_dict)
        else:
            fc_score, conv_score, loss = self.sess.run([self.model.fc_pred, self.model.conv_pred,
                                                              self.model.loss], feed_dict)

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

        return accu_add, accu_mul, loss, predictions_add, predictions_mul, gt_classes

