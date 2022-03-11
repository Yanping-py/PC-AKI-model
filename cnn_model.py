#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import auc_eval


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')

        self.cnn()

    def cnn(self):
        """cnn模型"""
        input_x = self.input_x

        with tf.name_scope("cnn"):

            fc1 = tf.layers.dense(input_x, self.config.hidden_dim, name='fc1')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc2 = tf.layers.dense(fc1, self.config.hidden_dim, name='fc2')
            fc2 = tf.contrib.layers.dropout(fc2,
                self.config.dropout_keep_prob)
            fc2 = tf.nn.relu(fc2)

            # 分类器
            self.logits = tf.layers.dense(fc2, self.config.num_classes,
                name='fc3')
            self.pred_y = tf.nn.softmax(self.logits)

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            ratio = 1/3
            class_weight = tf.constant([ratio,1-ratio])
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y,pos_weight=class_weight)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimize"):
            # 优化器
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)



        with tf.name_scope("Precison"):
            predicted = tf.argmax(self.pred_y,1)
            actuals = tf.argmax(self.input_y,1)
            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predicted = tf.ones_like(predicted)
            zeros_like_predicted = tf.zeros_like(predicted)

            tp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,ones_like_actuals),\
                tf.equal(predicted,ones_like_predicted)),tf.float32))
            tn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,zeros_like_actuals),\
                tf.equal(predicted,zeros_like_predicted)),tf.float32))
            fp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,zeros_like_actuals),\
                tf.equal(predicted,ones_like_predicted)),tf.float32))
            fn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals,ones_like_actuals),\
                tf.equal(predicted,zeros_like_predicted)),tf.float32))

            tpr = tf.div(tp_op, tp_op + fn_op)
            fpr = tf.div(fp_op, tp_op + fn_op)
            self.recall = tpr
            self.precision = tf.div(tp_op, tp_op + fp_op)
            self.specificity = tf.div(tn_op, tn_op +fp_op)
            self.npv = tf.div(tn_op,tn_op+fn_op)
            self.acc = tf.div(tp_op+tn_op, tp_op +tn_op+fp_op+fn_op)
            self.f1_score = (2*(self.precision*self.recall))/(self.precision + self.recall)

        with tf.name_scope('performance'):

            with tf.name_scope('auc'):
                label = tf.reshape(self.input_y, [-1, self.config.num_classes])
                pred = tf.reshape(self.pred_y, [-1, self.config.num_classes])
                self.auc = auc_eval.auc(labels=self.input_y, predictions=self.pred_y)
                tf.summary.scalar('macro_auc', tf.reduce_mean(self.auc))




  
