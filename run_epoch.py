#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from rnn_model import *
from cnn_model import *
from configuration import *
from data.aki_data_loader import *
import time
from datetime import timedelta

training_excel_path = " "
validation_excel_path = " "
testing_excel_path  = " "
mimic_excel_path  = " "


def run_epoch(cnn=True):
    # 载入数据
    print('Loading data...')
    start_time = time.time()
    x_train, y_train, x_test, y_test,x_valid, y_valid,x_mimic, y_mimic = \
        get_aki_data(training_excel_path, testing_excel_path,validation_excel_path,mimic_excel_path)
         
    if cnn:
        print('Using CNN model...')
        config = TCNNConfig()
        model = TextCNN(config)
        tensorboard_dir = 'D:/tensorboard/textcnn'
    else:
        print('Using RNN model...')
        config = TRNNConfig()
        model = TextRNN(config)
        tensorboard_dir = 'D:/tensorboard/textrnn'

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)

    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    initializer = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    session.run(initializer)

    # 配置 tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("specificity", model.specificity)
    tf.summary.scalar("recall", model.recall)
    tf.summary.scalar("precison", model.precision)
    tf.summary.scalar("npv", model.npv)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)
    

    def feed_data(batch):
        """准备需要喂入模型的数据"""
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch
        }
        return feed_dict, len(x_batch)

    def evaluate(x_, y_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        batch_eval = batch_iter(list(zip(x_, y_)), config.batch_size, 1)

        total_loss = 0.0
        total_specificity = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_npv = 0.0
        total_AUC = 0.0
        total_acc = 0.0
        cnt = 0
        for batch in batch_eval:
            feed_dict, cur_batch_len = feed_data(batch)
            loss, specificity,recall,precision,npv,acc = session.run([model.loss, model.specificity,model.recall, model.precision,model.npv,model.acc],
                feed_dict=feed_dict)
            auc = session.run(model.auc,feed_dict=feed_dict)
            total_loss += loss * cur_batch_len
            total_specificity += specificity * cur_batch_len
            total_recall += recall * cur_batch_len
            total_precision += precision * cur_batch_len
            total_npv += npv * cur_batch_len
            total_AUC += auc * cur_batch_len
            total_acc += acc * cur_batch_len
            cnt += cur_batch_len
        return total_loss / cnt, total_specificity / cnt, total_recall/ cnt, total_precision/cnt, total_npv / cnt,total_AUC / cnt,total_acc / cnt

    # 训练与验证
    print('Training and evaluating...')
    start_time = time.time()
    print_per_batch = config.print_per_batch
    for i, batch in enumerate(batch_train):
        feed_dict, _ = feed_data(batch)

        if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)

        if i % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
            loss_train, specificity_train,recall_train,precision_train,npv_train,acc_train= \
            session.run([model.loss, model.specificity,model.recall,model.precision,model.npv,model.acc], feed_dict=feed_dict)
            auc_train = session.run(model.auc, feed_dict = feed_dict)
            loss,specificity,recall,precision,npv,auc,acc = evaluate(x_valid,y_valid)
            
            # 时间
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))

           
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Specificity: {2:>7.2%},Train recall: {3:>7.2%}, Train precision: {4:>7.2%}, '\
                + 'Train auc: {5:>10.6},Valid Specificity:{6:>7.2%},Valid recall:{7:>7.2%},Valid precision:{8:>7.2%},Valid auc:{9:>10.6},Train acc:{10:>7.2%}'
            print(msg.format(i + 1, loss_train, specificity_train,recall_train,precision_train,auc_train,specificity,recall,precision,auc,acc_train))


        session.run(model.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    from tensorflow.python.framework import graph_util
   # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, ['score/Softmax'])

    #写入序列化的 PB 文件
    with tf.gfile.FastGFile('model18.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    loss_train, specificity_train,recall_train,precision_train,npv_train,auc_train,acc_train = evaluate(x_train, y_train)
    msg = 'train Loss: {0:>6.2}, train specificity: {1:>7.2%},train recall: {2:>7.2%}, train precision: {3:>7.2%}, train npv: {4:>7.2%},train_auc: {5:>10.6},train_acc:{6:>7.2%}'
    print(msg.format(loss_train, specificity_train,recall_train,precision_train,npv_train,auc_train,acc_train))

    loss_valid, specificity_valid,recall_valid,precision_valid,npv_valid,auc_valid,acc_valid = evaluate(x_valid, y_valid)
    msg = 'Valid Loss: {0:>6.2}, Valid specificity: {1:>7.2%},Valid recall: {2:>7.2%}, Valid precision: {3:>7.2%}, Valid npv: {4:>7.2%},Valid_auc: {5:>10.6},Valid_acc:{6:>7.2%}'
    print(msg.format(loss_valid, specificity_valid,recall_valid,precision_valid,npv_valid,auc_valid,acc_valid))

    loss_test, specificity_test,recall_test,precision_test,npv_test,auc_test,acc_test = evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test specificity: {1:>7.2%},Test recall: {2:>7.2%}, Test precision: {3:>7.2%}, Test npv: {4:>7.2%},Test auc: {5:>10.6},Test acc:{6:>7.2%}'
    print(msg.format(loss_test, specificity_test,recall_test,precision_test,npv_test,auc_test,acc_test))

    loss_mimic, specificity_mimic,recall_mimic,precision_mimic,npv_mimic,auc_mimic,acc_mimic = evaluate(x_mimic, y_mimic)
    msg = 'mimic Loss: {0:>6.2}, mimic specificity: {1:>7.2%},mimic recall: {2:>7.2%}, mimic precision: {3:>7.2%}, mimic npv: {4:>7.2%},mimic auc: {5:>10.6},mimic acc:{6:7.2%}'
    print(msg.format(loss_mimic, specificity_mimic,recall_mimic,precision_mimic,npv_mimic,auc_mimic,acc_mimic))
    session.close()

if __name__ == '__main__':
    run_epoch(cnn=True)
