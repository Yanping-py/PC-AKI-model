from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
from data.aki_data_loader import *
import auc_eval
from configuration import *
sess = tf.Session()
with gfile.FastGFile('dnn_model3.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图

training_excel_path = " "
validation_excel_path = " "
testing_excel_path  = " "
mimic_excel_path  = " "
x_train, y_train, x_test, y_test,x_valid, y_valid,x_mimic, y_mimic = \
        get_aki_data(training_excel_path, testing_excel_path,validation_excel_path,mimic_excel_path)
         
# 输入
input_x = sess.graph.get_tensor_by_name('input_x:0')
input_y1 = y_train
input_y2 = y_valid
input_y3 = y_test
input_y4 = y_mimic

op = sess.graph.get_tensor_by_name('score/Softmax:0')

predict_y1 = sess.run(op,  feed_dict={input_x: x_train})
predict_y2 = sess.run(op,  feed_dict={input_x: x_valid})
predict_y3 = sess.run(op,  feed_dict={input_x: x_test})
predict_y4 = sess.run(op,  feed_dict={input_x: x_mimic})
auc1 = auc_eval.auc(labels=input_y1, predictions=predict_y1)
auc2 = auc_eval.auc(labels=input_y2, predictions=predict_y2)
auc3 = auc_eval.auc(labels=input_y3, predictions=predict_y3)
auc4 = auc_eval.auc(labels=input_y4, predictions=predict_y4)


with tf.Session():  
    print(auc1.eval(),auc2.eval(),auc3.eval(),auc4.eval())
