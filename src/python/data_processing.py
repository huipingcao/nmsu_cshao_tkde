import numpy as np
import sys
import time
import os

########################################################################################
## data structure part
class data_structure:
    def __init__(self, num_classes, start_class, attr_num, attr_len, class_c=0):
        self.num_classes = num_classes
        self.start_class = start_class
        self.attr_num = attr_num
        self.attr_len = attr_len
        self.class_column = class_c

    def print_to_string(self):
        ret_str =  'num of classes: ' + str(self.num_classes) +'\nstart class: '+ str(self.start_class) +'\nattribute number: ' + str(self.attr_num) +'\nattribute length: ' + str(self.attr_len)  +'\nclass column: ' + str(self.class_column) 
        return ret_str


def return_data_stru(num_classes, start_class, attr_num, attr_len, class_column):
    return data_structure(num_classes, start_class, attr_num, attr_len, class_column)



def copy_data_stru(in_data_stru):
    return data_structure(in_data_stru.num_classes, in_data_stru.start_class, in_data_stru.attr_num, in_data_stru.attr_len)


## end of data structure part
########################################################################################



def train_test_transpose(data_matrix, attr_num, attr_len, trans=True):
    data_row, data_col = data_matrix.shape
    data_matrix = data_matrix.reshape(data_row, attr_num, attr_len, 1)
    #data_matrix = data_matrix.reshape(data_row, attr_num, 1, attr_len)
    if trans == True:
        data_matrix = np.transpose(data_matrix, (0, 2, 3, 1))
    else:
        data_matrix = np.transpose(data_matrix, (0, 2, 1, 3))
    #data_matrix = data_matrix.reshape(data_row, data_col)
    return data_matrix

def y_vector_to_matrix(y_vector, num_classes, start_class=0):
    vector_len = len(y_vector)
 #   print y_vector
 #   print vector_len
 #   print num_classes
 #   print "========"
    y_matrix = np.zeros((vector_len, num_classes))
    count = 0
    for item in y_vector:
        y_matrix[count, int(item)-start_class] = int(1)
        count = count + 1
    return y_matrix

def class_label_vector_checking(y_vector):
    min_class = min(y_vector)
    max_class = max(y_vector)
    class_index_dict = {}
    min_length = -1
    max_length = -1
    for c in range(min_class, max_class+1):
        c_index = np.where(y_vector==c)[0]
        class_index_dict[c] = c_index
        if min_length == -1:
            min_length = len(c_index)
        elif len(c_index) < min_length:
            min_length = len(c_index)
        if max_length == -1:
            max_length = len(c_index)
        elif len(c_index) > max_length:
            max_length = len(c_index)

    return class_index_dict, min_length, max_length


def feature_data_generation_4d(data_matrix, feature_index_list):
    row_n, attr_len, num_map, attr_num = data_matrix.shape
    
    ret_matrix = []
    new_row_col = 0
    
    new_attr = len(feature_index_list)

    new_row_col = new_attr * attr_len
    for i in range(0, row_n):
        ori_matrix = data_matrix[i].reshape(attr_len, attr_num)
        matrix = ori_matrix[:, feature_index_list]
        ret_matrix.append(matrix.reshape(new_row_col))
    
    data_matrix = np.array(ret_matrix).reshape(row_n, new_row_col)

    return np.array(ret_matrix).reshape(row_n, new_row_col), new_attr

def feature_data_generation(data_matrix, attr_len, attr_num, feature_index_list):
    row_n, col_n = data_matrix.shape
    ret_matrix = []
    new_row_col = 0
    
    new_attr = len(feature_index_list)

    new_row_col = new_attr * attr_len
    for i in range(0, row_n):
        ori_matrix = data_matrix[i].reshape(attr_len, attr_num)
        matrix = ori_matrix[:, feature_index_list]
        ret_matrix.append(matrix.reshape(new_row_col))
    
    data_matrix = np.array(ret_matrix).reshape(row_n, new_row_col)

    return np.array(ret_matrix).reshape(row_n, new_row_col), new_attr


def feature_data_generation_v1(data_matrix, attr_num, feature_index_list, group_list=[]):
    row_n, col_n = data_matrix.shape
    attr_len = col_n/attr_num
    ret_matrix = []
    new_row_col = 0
    
    new_attr = len(feature_index_list)

    if len(group_list) > 0:
        for group in group_list:
            new_attr = new_attr + len(group)

    new_row_col = new_attr * attr_len

    for i in range(0, row_n):
        ori_matrix = data_matrix[i].reshape(attr_num, attr_len)
        if len(group_list) > 0:
            group_count = 0
            for group in group_list:
                if group_count == 0:
                    matrix = ori_matrix[group, :]
                else:
                    matrix = np.append(matrix, ori_matrix[group, :])
                group_count = group_count + 1
            matrix = np.append(matrix, ori_matrix[feature_index_list, :])
        else:
            matrix = ori_matrix[feature_index_list, :]
        ret_matrix.append(matrix.reshape(new_row_col))
    
    data_matrix = np.array(ret_matrix).reshape(row_n, new_row_col)

    return np.array(ret_matrix).reshape(row_n, new_row_col), new_attr, attr_len


def z_normlization(time_series):
    mean = np.mean(time_series)
    dev = np.std(time_series)
    return (time_series - mean)/dev

