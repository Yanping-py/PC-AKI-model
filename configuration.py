#!/usr/bin/python
# -*- coding: utf-8 -*-

class TCNNConfig(object):
    """CNN配置参数"""

    # 模型参数
    seq_length = 14       # 序列长度79
    num_classes = 2        # 类别数
    num_filters = 256       # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    
    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 1.0 # dropout保留比例
    learning_rate = 2.00E-04  # 学习率1e-3

    batch_size = 4000        # 每批训练大小
    num_epochs = 2000          # 总迭代轮次2500

    print_per_batch = 10    # 每多少轮输出一次结果


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    seq_length = 79        # 序列长度
    num_classes = 2        # 类别数

    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 100          # 总迭代轮次
