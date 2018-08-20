from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import gc


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data_train, data_validate, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, config, logger)

        self.data_train = data_train
        self.data_validate = data_validate

    def train_epoch(self, curr_epoch):
        lr = self.config.basic_lr

        # todo: num_iter_per_epoch should be calaulated: (len_train_eff // params['batch_size']) + 1))
        loop_train = tqdm(range(self.config.num_train_iter_per_epoch))

        losses = []
        accs = []
        # iterate over steps (batches)
        for _ in loop_train:
            loss = self.train_validate_step(lr, True)

            cur_it = self.model.global_step_tensor.eval(self.sess)
            summaries_dict = {
                'loss': loss,
                #    'acc': acc,
            }
            self.logger.summarize(cur_it, summaries_dict=summaries_dict)

            #accs.append(acc)
            # todo: is this still necessary?
            gc.collect()

        # validate every few epochs
        if curr_epoch % self.config.val_interval == 0:
            # todo: num_iter_per_epoch should be calaulated: (len_train_eff // params['batch_size']) + 1))
            loop_validate = tqdm(range(self.config.num_val_iter_per_epoch))

            # iterate over steps (batches)
            for _ in loop_validate:
                loss = self.train_validate_step(lr, False)
                # todo: should i fix the cut_it?
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'loss_validation': loss,
                    #    'acc': acc,
                }
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)


        self.model.save(self.sess)

    def train_validate_step(self, lr, is_training):

        if is_training:
            batch_fc_img, batch_conv_img, batch_labels = self.data_train.get_next_batch()
            prob = 0.5
        else:
            batch_fc_img, batch_conv_img, batch_labels = self.data_validate.get_next_batch()
            prob = 1.0

        feed_dict = {
            self.model.fc_img: batch_fc_img,
            self.model.conv_img: batch_conv_img,
            self.model.ys: batch_labels,
            self.model.Lr: lr,
            self.model.is_training: is_training,
            self.model.prob: prob
        }

        # todo: add accuracy and different types of loss/ alphas - create designated functions
        if is_training:
            _, loss = self.sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)
        else:
            loss = self.sess.run([self.model.loss], feed_dict=feed_dict)

        return loss




# iterate over epochs
for l in range(1000):




        summary, test_fc_score, test_conv_score, test_loss = compute_score_loss(full_model_specs, sess, batch_fc_img,
                                                                                batch_conv_img, batch_labels)

        writer_test.add_summary(summary, ((n_test_examples // params['batch_size'] + 1) * l + i))

        print("test_loss:", test_loss)
        test_fc_score = np.reshape(np.array(test_fc_score),
                                   (params['batch_size'], params['n_classes']))  # (batch_size, n_classes)
        test_conv_score = np.reshape(np.array(test_conv_score), (params['batch_size'], params['n_classes']))

        fc_scores.append(test_fc_score)
        conv_scores.append(test_conv_score)
        pbar.update(1)

    fc_scores = np.reshape(np.array(fc_scores), (-1, params['n_classes']))[:n_test_examples]
    conv_scores = np.reshape(np.array(conv_scores), (-1, params['n_classes']))[:n_test_examples]

    print(fc_scores.shape)
    print(conv_scores.shape)

    pbar.close()

    num_test_score = fc_scores
    test_label_pred = np.argmax(num_test_score, axis=1)
    print(test_label_pred.shape)
    print(l, "epoch:, attention，Fc_lstm_ACC:", accuracy(test_label_pred, test_examples_labels))
    test_info = []
    for i in range(n_test_examples):
        video_info = []
        video_info.append(num_test_score[i])
        video_info.append(test_examples_labels[i])
        test_info.append(video_info)

    npz_name = "score_attention/" + str(l) + "_epoch_fc_score.npz"
    np.savez(npz_name, test_info=test_info)

    num_test_score = conv_scores
    test_label_pred = np.argmax(num_test_score, axis=1)
    print(test_label_pred.shape)
    print(l, "epoch:, attention，Conv_lstm_ACC:", accuracy(test_label_pred, test_examples_labels))
    test_info = []
    for i in range(n_test_examples):
        video_info = []
        video_info.append(num_test_score[i])
        video_info.append(test_examples_labels[i])
        test_info.append(video_info)

    npz_name = "score_attention/" + str(l) + "_epoch_conv_score.npz"
    np.savez(npz_name, test_info=test_info)

    num_test_score = np.add(fc_scores, conv_scores)
    test_label_pred = np.argmax(num_test_score, axis=1)
    print(test_label_pred.shape)
    print(l, "epoch:, attention，Add_fusion_ACC:", accuracy(test_label_pred, test_examples_labels))
    test_info = []
    for i in range(n_test_examples):
        video_info = []
        video_info.append(num_test_score[i])
        video_info.append(test_examples_labels[i])
        test_info.append(video_info)

    npz_name = "score_attention/" + str(l) + "_epoch_numscore.npz"
    np.savez(npz_name, test_info=test_info)

    mul_test_score = np.multiply(fc_scores, conv_scores)
    test_label_pred = np.argmax(mul_test_score, axis=1)
    print(test_label_pred.shape)
    print(l, "epoch:, attention，Mul_fusion_ACC:", accuracy(test_label_pred, test_examples_labels))
    test_info = []
    for i in range(n_test_examples):
        video_info = []
        video_info.append(mul_test_score[i])
        video_info.append(test_examples_labels[i])
        test_info.append(video_info)

    npz_name = "score_attention/" + str(l) + "_epoch_mulscore.npz"
    np.savez(npz_name, test_info=test_info)

    # reset test data feeder
    test_feeder.reset_feeder()
