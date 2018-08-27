import os

import tensorflow as tf

from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(labels, correct_labels, predict_labels, title='Confusion matrix', tensor_name='MyFigure/image', normalize=False):

# Parameters:
#     correct_labels                  : These are your true classification categories.
#     predict_labels                  : These are you predicted classification categories
#     labels                          : This is a lit of labels which will be used to display the axix labels
#     title='Confusion matrix'        : Title for your matrix
#     tensor_name = 'MyFigure/image'  : Name for the output summay tensor
#
# Returns:
#     summary: TensorFlow summary
#
# Other itema to note:
#     - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
#     - Currently, some of the ticks dont line up due to rotations.

    cm = confusion_matrix(correct_labels, predict_labels, labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(3, 3), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=4)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=2, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=4)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=2, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=3, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


class Logger:
    def __init__(self, sess,config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))

        self.img_d_summary_dir = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "val_img"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]),
                                                                            name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)

                if hasattr(self,'experiment') and self.experiment is not None:
                    self.experiment.log_multiple_metrics(summaries_dict, step=step)

                summary_writer.flush()

    def confusion_mat(self, step, labels, gt_classes, predictions_add, predictions_mul, mode):
        conf_mat_add_summary = plot_confusion_matrix(labels, gt_classes, predictions_add, title="Confusion matrix- Add", tensor_name='MyFigure/add')
        conf_mat_mul_summary = plot_confusion_matrix(labels, gt_classes, predictions_mul, title="Confusion matrix- Multiply", tensor_name='MyFigure/mul')

        if mode == 'train':
            summary_writer = self.train_summary_writer
        elif mode == 'validate':
            summary_writer = self.img_d_summary_dir
        else:
            summary_writer = self.test_summary_writer

        summary_writer.add_summary(conf_mat_add_summary, step)
        summary_writer.add_summary(conf_mat_mul_summary, step)
        summary_writer.flush()


