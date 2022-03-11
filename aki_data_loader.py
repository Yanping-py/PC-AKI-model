from collections import Counter
import numpy as np
import os
import xlrd

def _read_excel(filename):
    """读取excel数据"""
    num_var = 14
    input_features = []
    labels = []
    data = xlrd.open_workbook(filename)
    sheet = data.sheet_by_index(0)
    nrows = sheet.nrows
    ncols = sheet.ncols
    for row_index in range(1,nrows):
        feature = []
        label = []
        for col_index in range(0,ncols-1):
            feature.append(float(sheet.cell_value(row_index,col_index)))
        if sheet.cell_value(row_index,num_var) == 1:
            label = [0.0, 1.0]
        else:
            label = [1.0, 0.0]
        input_features.append(feature)
        labels.append(label)
    return input_features, labels

def get_aki_data(training_excel_path, testing_excel_path,validation_excel_path,mimic_excel_path):
    input_training_features, training_labels = _read_excel(training_excel_path)
    input_validation_features, validation_labels = _read_excel(validation_excel_path)
    input_testing_features, testing_labels = _read_excel(testing_excel_path)
    input_mimic_features, mimic_labels = _read_excel(mimic_excel_path)
    return input_training_features, training_labels, input_testing_features, testing_labels,input_validation_features, \
        validation_labels,input_mimic_features, mimic_labels
    

def batch_iter(data, batch_size=64, num_epochs=5):
    """生成批次数据"""
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        indices = np.arange(data_size)
        shuffled_data = data[indices]

        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

    