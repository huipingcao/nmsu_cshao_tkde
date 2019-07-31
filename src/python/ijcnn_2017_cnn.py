import numpy as np
import sys
import time
import gc
import os

import tensorflow as tf

from log_io import init_logging
from object_io import save_obj

from tensorflow_cnn import cnn_train
from tensorflow_cnn import conf_conv_layer
from tensorflow_cnn import conf_pool_layer
from tensorflow_cnn import conf_out_layer
from data_processing import return_data_stru
from data_processing import y_vector_to_matrix
from model_setting import return_cnn_setting_from_file
from classification_results import f1_value_precision_recall_accuracy
from classification_results import predict_matrix_with_prob_to_predict_accuracy

def run_feature_projected_ijcnn_fcn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, data_stru, cnn_setting, feature_dict, top_k, saver_file_profix='', class_id=-1, logger=None):
    if logger == None:
        logger = init_logging('')

    real_num_classes = data_stru.num_classes
    data_stru.num_classes = 2
    cnn_setting.num_classes = 2
    num_classes = 2
    method = 'fcn'
    train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    test_row, attr_len, attr_num, input_map = test_x_matrix.shape

    all_predict_matrix = np.zeros(test_row * real_num_classes).reshape(test_row, real_num_classes)

    saver_file = ''
    if class_id == -1:
        min_class = min(train_y_vector)
        min_class = 8
        max_class = max(train_y_vector) + 1
    else:
        min_class = class_id
        max_class = class_id + 1

    saver_file_profix = saver_file_profix + '_class'

    keep_saver_file = ''
    all_train_time = 0
    all_test_time = 0
    all_f1_value = []
    all_train_time = []
    all_test_time = []
    for i in range(min_class, max_class):
        logger.info('class: ' + str(i))
        temp_train_y_vector = np.where(train_y_vector == i, 1, 0)
        temp_test_y_vector = np.where(test_y_vector == i, 1, 0)
        class_saver_profix = saver_file_profix + str(i)

        fold_positive_len = len(np.where(temp_train_y_vector == 1)[0])
        fold_negative_len = len(temp_train_y_vector) - fold_positive_len

        logger.info("=====")
        logger.info("positive class labels length: " + str(fold_positive_len))
        logger.info("negative class labels length: " + str(fold_negative_len))
        if feature_dict == None:
            temp_train_x_matrix = train_x_matrix
            temp_test_x_matrix = test_x_matrix
        else:
            class_feature = feature_dict[i]
            class_feature = class_feature[0:top_k]
            print top_k
            print class_feature
            logger.info("feature list: " + str(class_feature))
            temp_train_x_matrix = train_x_matrix[:, :, class_feature, :]
            temp_test_x_matrix = test_x_matrix[:, :, class_feature, :]
        temp_train_y_matrix = y_vector_to_matrix(temp_train_y_vector, num_classes)
        temp_test_y_matrix = y_vector_to_matrix(temp_test_y_vector, num_classes)

        if i == min_class:
            input_x_placeholder = tf.placeholder(tf.float32, [None, attr_len, attr_num, input_map])
            output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
            predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file = fcn_configure(input_x_placeholder, num_classes, logger)
            keep_saver_file = saver_file

        saver_file = class_saver_profix + keep_saver_file
        
        class_eval_value, class_train_time, class_test_time, class_predict_prob, fold_saver_file, fold_obj_file = cnn_train(
        temp_train_x_matrix, temp_train_y_matrix, temp_test_x_matrix, temp_test_y_matrix, num_classes, cnn_setting, input_x_placeholder, output_y_placeholder, predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
        
        class_predict_y = np.argmax(class_predict_prob, axis=1)
        class_accuracy, precision, recall, class_f1, tp, fp, tn, fn = f1_value_precision_recall_accuracy(class_predict_y, temp_test_y_vector, 1)
        if str(class_eval_value) == 'nan':
            class_eval_value = 0
            class_f1 = 0
        logger.info(method + " f1 for class "+ str(i) + ": " + str(class_f1))
        logger.info(method + " accuracy for class "+ str(i) + ": " + str(class_accuracy))
        logger.info(method + ' model saved: ' + fold_saver_file)
        all_f1_value.append(class_f1)
        all_train_time.append(class_train_time)
        all_test_time.append(class_test_time)
        all_predict_matrix[:, i] = class_predict_prob[:, 1]
        #if i > 2:
        #    break
    all_accuracy, all_predict_y = predict_matrix_with_prob_to_predict_accuracy(all_predict_matrix, test_y_vector)
    return all_accuracy, all_f1_value, all_predict_y, all_train_time, all_test_time, all_predict_matrix



#train_x_matrix: train_row, attr_len, attr_num, input_map
#test_x_matrix: test_row, attr_len, attr_num, input_map
def run_ijcnn_fcn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, cnn_setting, saver_file_profix='', logger=None):
    if logger == None:
        logger = init_logging('')
    start_class = 0
    class_column = 0
    train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    cnn_setting.feature_method = 'none'

    num_classes = train_y_matrix.shape[1]
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    input_x_placeholder = tf.placeholder(tf.float32, [None, attr_len, attr_num, input_map])
    output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
    predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file = fcn_configure(input_x_placeholder, num_classes, logger)

    saver_file = saver_file_profix + saver_file
    cnn_eval_value, train_run_time, test_run_time, cnn_predict_prob, saver_file, feature_list_obj_file = cnn_train(
        train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, input_x_placeholder, output_y_placeholder, predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
    if str(cnn_eval_value) == 'nan':
        cnn_eval_value = 0
    return cnn_eval_value, train_run_time, test_run_time, cnn_predict_prob, saver_file, feature_list_obj_file
    #return cnn_accuracy, train_run_time, test_run_time, cnn_predict_prob, train_sensor_result, weight_fullconn, bias_fullconn




def fcn_configure(input_x_placeholder, num_classes, logger):
    if logger == None:
        logger = init_logging('')
    #conv_kernel_list = [[1,8], [1,5], [1,3]]
    conv_kernel_list = [[8, 1], [5, 1], [3, 1]]
    conv_kernel_list = np.array(conv_kernel_list)
    feature_num_list = [50, 40, 20]
    activation_fun = 0
    num_input_map = 1
    conv_row_num = len(conv_kernel_list)
    saver_file = ''
    strides_list = [1, 1, 1, 1]
    std_value = 0.02
    same_size = False
    out_conv = input_x_placeholder
    keeped_feature_list = []
    for i in range(0, conv_row_num):
        logger.info('layer: ' + str(i) + " input:")
        logger.info(out_conv.get_shape())
        conv_row_kernel = conv_kernel_list[i, 0]
        conv_col_kernel = conv_kernel_list[i, 1]

        num_output_map = feature_num_list[i]

        saver_file = saver_file + "_c" + str(conv_row_kernel) + "_" + str(conv_col_kernel)
        out_conv = conf_conv_layer(i, conv_row_kernel, conv_col_kernel, out_conv, num_input_map, num_output_map, activation_fun, strides_list, std_value, same_size)
        logger.info("Conv output: " + str(out_conv.get_shape()))
        out_conv = tf.layers.batch_normalization(out_conv)
        logger.info("Conv after batch normal: " + str(out_conv.get_shape()))
        num_input_map = num_output_map

    row_samp_rate = out_conv.get_shape()[1]
    col_samp_rate = 1
    out_conv = conf_pool_layer(out_conv, row_samp_rate, col_samp_rate, False)
    keeped_feature_list.append(out_conv)
    logger.info("Feature result shape")
    logger.info(out_conv.get_shape())

    saver_file = saver_file + "global_p" + str(row_samp_rate) + "_" + str(col_samp_rate) + '.ckpt'

    #dropout
    keep_prob_placeholder = tf.placeholder(tf.float32)
    out_conv = tf.nn.dropout(out_conv, keep_prob_placeholder)

    out_fir, out_sec, out_thi, out_for = out_conv.get_shape()
    feature_num = int(out_sec * out_thi * out_for)
    print out_conv.get_shape()
    out_conv = tf.reshape(out_conv, [-1, feature_num])
    print std_value
    print feature_num
    predict_y_prob = conf_out_layer(out_conv, feature_num, num_classes, std_value)
    #print "predict_y_prob"
    print predict_y_prob.get_shape()
    return predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file
# End of CNN method


if __name__ == '__main__':
    cnn_setting_file = "../../parameters/cnn_model_parameter.txt"
    cnn_setting = return_cnn_setting_from_file(cnn_setting_file)

    train_row = 20
    test_row = 10
    num_classes = 3
    attr_num =45
    attr_len = 125
    data_stru = return_data_stru(num_classes, 0, attr_num, attr_len, 0)
    train_x_matrix = np.random.rand(train_row, attr_len, attr_num, 1)
    test_x_matrix = np.random.rand(test_row, attr_len, attr_num, 1)
    train_y_vector = np.array([0,0,0,0,0,1,1,1,1,1,1,2,2,2,1,1,0,0,0,2])
    test_y_vector = np.array([0,0,0,1,1,1,0,0,2,2])
    train_y_matrix = y_vector_to_matrix(train_y_vector, num_classes)
    test_y_matrix = y_vector_to_matrix(test_y_vector, num_classes)
    print train_x_matrix.shape
    print train_y_matrix.shape
    print test_x_matrix.shape
    print test_y_matrix.shape
    run_feature_projected_ijcnn_fcn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, data_stru, cnn_setting, None, 1, '', -1, None)
    
    
