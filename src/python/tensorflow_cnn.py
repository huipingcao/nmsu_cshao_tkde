import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
#from tensorflow.contrib.data.python.ops import sliding
import numpy as np
import sys
import time
import gc
import os
from math import sqrt

from data_processing import feature_data_generation
from data_processing import y_vector_to_matrix
from data_processing import return_data_stru
from data_processing import copy_data_stru

from model_setting import return_cnn_default_setting

from log_io import init_logging
from log_io import setup_logger
from object_io import save_obj
from data_processing import class_label_vector_checking




def conf_act(input_conv, activation_fun=0, relu_base=None):
    tf.random.set_random_seed(0)
    if activation_fun == 0:
        #ret_conv = tf.nn.relu(input_conv)
        comparison = tf.less(input_conv, tf.constant(0.0))
        negative_one = positive_one - 2
        relu_weight = tf.where(comparison, negative_one, positive_one)
        if relu_base is None:
            ret_conv = tf.nn.relu(input_conv)
            relu_base = relu_weight
        else:
            relu_base = relu_base + relu_weight
            comparison = tf.less(relu_base, tf.constant(0.0))
            positive_one = tf.ones_like(relu_base)
            overall_weight = tf.where(comparison, tf.zeros_like(relu_base), positive_one)
            ret_conv = tf.math.multiply(input_conv, overall_weight) 
    elif activation_fun == 1:
        ret_conv = tf.nn.sigmoid(input_conv)
    elif activation_fun == 2:
        ret_conv = tf.nn.tanh(input_conv)
    elif activation_fun == 3:
        ret_conv = tf.nn.relu(input_conv)
    ####
    # 4: relu for the attribute level. if the overall negative values for this attribute is more than positive ones. this attribute will be abandoned
    # for other attributes, regular relu will be applied
    elif activation_fun == 4:
        input_shape = input_conv.get_shape()
        attr_len = int(input_shape[1])
        if len(input_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            comparison = tf.less(input_conv, tf.constant(0.0))
            positive_one = tf.ones_like(input_conv)
            negative_one = positive_one - 2
            relu_weight = tf.where(comparison, negative_one, positive_one)
            relu_weight_sum = tf.reduce_sum(relu_weight, 1, keep_dims=True)
            comparison = tf.less(relu_weight_sum, tf.constant(0.1))
            posotive_vector = tf.ones_like(relu_weight_sum)
            zero_vector = tf.zeros_like(relu_weight_sum)
            relu_vector = tf.where(comparison, zero_vector, posotive_vector)
            #print relu_vector.get_shape()
            relu_matrix = tf.tile(relu_vector, [1, attr_len, 1, 1])
            #print relu_matrix.get_shape()
            ret_conv = tf.math.multiply(input_conv, relu_matrix)
    elif activation_fun == 5:
        input_shape = input_conv.get_shape()
        attr_len = int(input_shape[1])
        if len(input_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            comparison = tf.less(input_conv, tf.constant(0.0))
            positive_one = tf.ones_like(input_conv)
            negative_one = positive_one - 2
            relu_weight = tf.where(comparison, negative_one, positive_one)
            relu_weight_sum = tf.reduce_sum(relu_weight, 1, keep_dims=True)
            comparison = tf.less(relu_weight_sum, tf.constant(0.1))
            posotive_vector = tf.ones_like(relu_weight_sum)
            zero_vector = tf.zeros_like(relu_weight_sum)
            relu_vector = tf.where(comparison, zero_vector, posotive_vector)
            #print relu_vector.get_shape()
            relu_matrix = tf.tile(relu_vector, [1, attr_len, 1, 1])
            #print relu_matrix.get_shape()
            ret_conv = tf.math.multiply(input_conv, relu_matrix)
            ret_conv = tf.nn.relu(ret_conv)
    elif activation_fun == 6:
        input_shape = input_conv.get_shape()
        attr_len = int(input_shape[1])
        if len(input_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            attr_num = int(input_shape[2])
            input_row_sum = tf.reduce_sum(input_conv, 2, keep_dims=True)
            comparison = tf.less(input_row_sum, tf.constant(0.0))
            zero_vector = tf.zeros_like(input_row_sum)
            one_vector = tf.ones_like(input_row_sum)
            relu_vector = tf.where(comparison, zero_vector, one_vector)
            relu_matrix = tf.tile(relu_vector, [1, 1, attr_num, 1])
            ret_conv = tf.multiply(input_conv, relu_matrix)
    elif activation_fun == 7:
        input_shape = input_conv.get_shape()
        attr_len = int(input_shape[1])
        if len(input_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            attr_num = int(input_shape[2])
            input_row_sum = tf.reduce_sum(input_conv, 2, keep_dims=True)
            comparison = tf.less(input_row_sum, tf.constant(0.0))
            zero_vector = tf.zeros_like(input_row_sum)
            one_vector = tf.ones_like(input_row_sum)
            relu_vector = tf.where(comparison, zero_vector, one_vector)
            relu_matrix = tf.tile(relu_vector, [1, 1, attr_num, 1])
            ret_conv = tf.multiply(input_conv, relu_matrix)
            ret_conv = tf.nn.relu(ret_conv)

    return ret_conv, relu_base



def conf_conv_layer(layer, kernel_r, kernel_c, input_matrix, num_input_map, num_output_map, activation_fun=0, strides_list=[1,1,1,1], std_value=0.1, same_size=False, relu_base=None, logger=None):
    #if layer == 0:
    #    std_value = sqrt(0.2)
    #else:
    #    std_value = sqrt(0.2 / num_input_map)
    if logger is None:
        logger = setup_logger("")
    tf.random.set_random_seed(layer)
    weight_variable = tf.Variable(tf.truncated_normal([kernel_r, kernel_c, num_input_map, num_output_map], stddev=std_value), name='conv_w_'+str(layer))
    
    bias_variable = tf.Variable(tf.constant(std_value, shape=[num_output_map]), name='conv_b_'+str(layer))
    #bias_variable = tf.Variable(tf.constant(std_value, shape=[num_output_map]))
    #weight_variable = tf.Variable(tf.truncated_normal([kernel_r, kernel_c, num_input_map, num_output_map], stddev=std_value), name='conv_weight_'+str(layer))
    #bias_variable = tf.Variable(tf.constant(0.0, shape=[num_output_map]), name='conv_bias_'+str(layer))
    if same_size == "True":
        str_padding = 'SAME'
    else:
        str_padding = 'VALID'

    ret_conv_before_act = tf.nn.conv2d(input_matrix, weight_variable, strides=[1, 1, 1, 1], padding=str_padding) + bias_variable

    ret_conv, relu_base = conf_act(ret_conv_before_act, activation_fun, relu_base)
    #if activation_fun == 0:
    #    ret_conv = tf.nn.relu(ret_conv_before_act)
    #elif activation_fun == 1:
    #    ret_conv = tf.nn.sigmoid(ret_conv_before_act)
    #elif activation_fun == 2:
    #    ret_conv = tf.nn.tanh(ret_conv_before_act)
    ##elif activation_fun == 3:
#
    #elif activation_fun == -1:
    #    ret_conv = ret_conv_before_act
    #return ret_conv, ret_conv_before_act, weight_variable, bias_variable
    return ret_conv, relu_base


def cnn_set_flow_graph(data_stru, cnn_setting, input_map, group_all=False, logger=None):
    if logger == None:
        logger = init_logging('')
    tf.reset_default_graph()
    tf.random.set_random_seed(0)

    attr_num = data_stru.attr_num
    attr_len = data_stru.attr_len
    num_classes = data_stru.num_classes

    output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
    train_x_placeholder = tf.placeholder(tf.float32, [None, attr_len, attr_num, input_map])
    predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, relu_base_array = cnn_configure(
        train_x_placeholder, cnn_setting, num_classes, group_all, logger)
    return train_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, relu_base_array



############################################################################################################################################


#train_x_matrix: train_row, attr_len, attr_num, input_map
#test_x_matrix: test_row, attr_len, attr_num, input_map
def run_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, group_all=False, saver_file_profix='', logger=None):
    if logger == None:
        logger = init_logging('')
    num_classes = data_stru.num_classes
    attr_num = data_stru.attr_num
    attr_len = data_stru.attr_len
    logger.info(cnn_setting)

    if len(train_x_matrix.shape) == 3:
        train_row, attr_len, attr_num = train_x_matrix.shape
        input_map = 1
        train_x_matrix = train_x_matrix.reshape(train_row, attr_len, attr_num, input_map)
        test_row, attr_len, attr_num = test_x_matrix.shape
        test_x_matrix = test_x_matrix.reshape(test_row, attr_len, attr_num, input_map)
    else:
        train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    data_stru.attr_num = attr_num
    data_stru.attr_len = attr_len

    train_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, relu_base_array = cnn_set_flow_graph(data_stru, cnn_setting, input_map, group_all, logger)

    saver_file = saver_file_profix + "_group_" + str(group_all) + saver_file
    cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file = cnn_train(
        train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, train_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
    if str(cnn_eval_value) == 'nan':
        cnn_eval_value = 0
    return cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file, relu_base_array
    #return cnn_accuracy, train_run_time, test_run_time, cnn_predict_proba, train_sensor_result, weight_fullconn, bias_fullconn


#train_x_matrix: train_row, attr_len, attr_num, input_map
#test_x_matrix: test_row, attr_len, attr_num, input_map
def run_projected_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, group_all=False, saver_file_profix='', logger=None):
    if logger == None:
        logger = init_logging('')
    num_classes = data_stru.num_classes
    attr_num = data_stru.attr_num
    attr_len = data_stru.attr_len
    logger.info(cnn_setting)

    train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    data_stru.attr_num = attr_num
    data_stru.attr_len = attr_len

    train_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_set_flow_graph(
        data_stru, cnn_setting, input_map, group_all, logger)

    saver_file = saver_file_profix + "_group_" + str(group_all) + saver_file
    cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file = cnn_train(
        train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, train_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
    if str(cnn_eval_value) == 'nan':
        cnn_eval_value = 0
    return cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file
    #return cnn_accuracy, train_run_time, test_run_time, cnn_predict_proba, train_sensor_result, weight_fullconn, bias_fullconn


# bc_type: the type of batch controlled
# bc_type is 0: The proposed batch controlled method
# bc_type is 1: the normal batch weight 
# rand_class_index_dict: contains the indexes for all the class instances
def batch_controlled(iter, batch_x_matrix, batch_y_matrix, min_class, max_class, logger=None, batch_each_class=0, rand_train_class_index_dict=None, rand_train_class_start_dict=None, bc_type=0, func_key="batch_controlled"):
    if bc_type == 1:
        if logger is None:
            logger = setup_logger('')
        ## Normal BATCH Weight
        batch_y_vector = np.argmax(batch_y_matrix, axis=1)
        batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
        coefficients_vector = []
        batch_class_index_dict_keys = batch_class_index_dict.keys()
        for c_label in range(min_class, max_class+1):
            if c_label not in batch_class_index_dict_keys:
                add_index_vector_len = 0.1
            else:
                add_index_vector_len = len(batch_class_index_dict[c_label])
            coefficients_vector.append(float(batch_max_length)/float(add_index_vector_len))
        coefficients_vector = np.array(coefficients_vector)
        ## End of Normal BATCH Weight
    elif bc_type == 0:
        if batch_each_class <= 0 or rand_train_class_index_dict is None:
            logger.info(func_key + " function inputs are wrong!!!")
            raise Exception(func_key + " function inputs are wrong!!!")
            return
        batch_y_vector = np.argmax(batch_y_matrix, axis=1)
        batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
        coefficients_vector = []
        batch_class_index_dict_keys = batch_class_index_dict.keys()
        for c_label in range(min_class, max_class+1):
            if c_label not in batch_class_index_dict_keys:
                c_lable_start = rand_train_class_start_dict[c_label]
                
                c_label_index = rand_train_class_index_dict[c_label]
                c_label_index_len = len(c_label_index)
                add_index_vector_len = 0
                if c_label_index_len > batch_each_class:
                    add_index_vector = c_label_index
                    add_index_vector = np.random.choice(c_label_index_len, batch_each_class, replace=False)
                    if (i<3):
                        logger.info("add index vector for c " + str(c_label))
                        logger.info(add_index_vector)
                    add_index_vector_len = len(add_index_vector)
                    batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                    batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                else:
                    batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                    batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                    add_index_vector_len = c_label_index_len
            else:
                batch_class_index = batch_class_index_dict[c_label]
                add_index_vector_len = len(batch_class_index)
                c_label_index = train_class_index_dict[c_label]
                c_label_index_len = len(c_label_index)
                if add_index_vector_len < batch_each_class:
                    add_count = batch_each_class - add_index_vector_len
                    if c_label_index_len > add_count:
                        add_index_vector = np.random.choice(c_label_index_len, add_count, replace=False)
                        if (i<3):
                            logger.info("add index vector for c " + str(c_label))
                            logger.info(add_index_vector)
                        add_index_vector_len = add_index_vector_len + len(add_index_vector)
                        batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                        batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                    else:
                        batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                        batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                        add_index_vector_len = add_index_vector_len + c_label_index_len
                elif add_index_vector_len > 2 * batch_each_class:
                    remove_count = (add_index_vector_len - 2 * batch_each_class)
                    remove_index_vector = np.random.choice(batch_class_index, remove_count, replace=False)
                    add_index_vector_len = add_index_vector_len - len(remove_index_vector)
                    batch_x_matrix = np.delete(batch_x_matrix, remove_index_vector, axis=0)
                    batch_y_matrix = np.delete(batch_y_matrix, remove_index_vector, axis=0)
                    batch_y_vector = np.argmax(batch_y_matrix, axis=1)
                    batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
            coefficients_vector.append(float(add_index_vector_len))
        #print "End of F1"
        
            
        coefficients_vector = np.array(coefficients_vector)
        batch_max_len = float(max(coefficients_vector))
        coefficients_vector = batch_max_len/coefficients_vector
        if i < 3:
            batch_y_vector = np.argmax(batch_y_matrix, axis=1)
            batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
            logger.info("class index after: ")
            logger.info(batch_class_index_dict)
            logger.info("coefficient vector: ")
            logger.info(coefficients_vector)


def cnn_train(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, input_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob, keeped_feature_list, saver_file="./", logger=None):
    if logger == None:
        logger = init_logging('')
    min_class = 0
    eval_method = cnn_setting.eval_method
    batch_size = cnn_setting.batch_size
    stop_threshold = cnn_setting.stop_threshold
    max_iter = cnn_setting.max_iter
    feature_method = cnn_setting.feature_method
    feature_obj_file = cnn_setting.out_obj_folder + saver_file
    saver_file = cnn_setting.out_model_folder + saver_file
    #print predict_y_proba.get_shape()
    #print output_y_placeholder.get_shape()
    #print "======" 
    prediction = tf.argmax(predict_y_proba, 1)
    actual = tf.argmax(output_y_placeholder, 1)
    correct_prediction = tf.equal(prediction, actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if eval_method=='f1':
        train_y_vector = np.argmax(train_y_matrix, axis=1)
        train_class_index_dict, train_min_length, train_max_length = class_label_vector_checking(train_y_vector)
        min_class = 0
        max_class = max(train_y_vector)
        num_classes = max_class + 1
        if max_class == 1:
            TP = tf.count_nonzero(prediction * actual, dtype=tf.float32)
            TN = tf.count_nonzero((prediction - 1) * (actual - 1), dtype=tf.float32)
            FP = tf.count_nonzero(prediction * (actual - 1), dtype=tf.float32)
            FN = tf.count_nonzero((prediction - 1) * actual, dtype=tf.float32)
            precision = (TP) / (TP + FP)
            recall = (TP) / (TP + FN)
            f1 = (2 * precision * recall) / (precision + recall)
            eval_method_value = f1
            eval_method_keyword = "f1"
        else:
            eval_method_value = accuracy
            eval_method_keyword = "acc with batch"
        coefficient_placeholder = tf.placeholder(tf.float32, shape=[num_classes])
        cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=output_y_placeholder, logits=predict_y_proba, pos_weight=coefficient_placeholder))
    else:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_y_placeholder, logits=predict_y_proba))
        eval_method_value = accuracy
        eval_method_keyword = "acc"
    #print cross_entropy.get_shape()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    #tf.random.set_random_seed(0)
    cnn_session = tf.InteractiveSession()
    cnn_session.run(tf.global_variables_initializer())
    
    test_eval_value = 0
    best_eval_value = 0
    i = 0
    start = 0
    epoch = 0
    end = batch_size
    batch_each_class = int(batch_size/num_classes)
    overall_len = len(train_y_matrix)
    saver = tf.train.Saver()
    train_run_time = 0
    np.random.seed(epoch)
    batch_index = np.random.permutation(overall_len)
    logger.info("Random Epoch:" + str(epoch) + str(batch_index[0:5]))
    f1_unbalance_count = np.zeros(num_classes)
    second_chance = False
    re_init = False
    while(test_eval_value < stop_threshold):
        if start >= overall_len:
            start = 0
            end = start + batch_size
            epoch = epoch + 1
            np.random.seed(epoch)
            logger.info("Random Epoch:" + str(epoch) + str(batch_index[0:5]))
            batch_index = np.random.permutation(overall_len)
        elif end > overall_len:
            end = overall_len
        batch_x_matrix = train_x_matrix[batch_index[start:end], :, :, :]
        batch_y_matrix = train_y_matrix[batch_index[start:end], :]


        #print 'batch_x_matrix shape'
        #print batch_x_matrix.shape
        #print batch_y_matrix.shape
        if eval_method == 'f1':
            if i == 0:
                logger.info("Batch controlled")
            ### Normal BATCH Weight
            #batch_y_vector = np.argmax(batch_y_matrix, axis=1)
            #batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
            #coefficients_vector = []
            #batch_class_index_dict_keys = batch_class_index_dict.keys()
            #for c_label in range(min_class, max_class+1):
            #    if c_label not in batch_class_index_dict_keys:
            #        add_index_vector_len = 0.1
            #    else:
            #        add_index_vector_len = len(batch_class_index_dict[c_label])
            #    coefficients_vector.append(float(batch_max_length)/float(add_index_vector_len))
            #coefficients_vector = np.array(coefficients_vector)
            ### End of Normal BATCH Weight
            # BATCH_CONTROLLED
            batch_y_vector = np.argmax(batch_y_matrix, axis=1)
            batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
            if i < 3:
                logger.info("class index before: ")
                logger.info(batch_class_index_dict)
            coefficients_vector = []
            batch_class_index_dict_keys = batch_class_index_dict.keys()
            for c_label in range(min_class, max_class+1):
                #print "class: " + str(c_label)
                #print class_label_vector_checking
                if c_label not in batch_class_index_dict_keys:
                    f1_unbalance_count[c_label] = f1_unbalance_count[c_label] + 1
                    c_label_index = train_class_index_dict[c_label]
                    c_label_index_len = len(c_label_index)
                    add_index_vector_len = 0
                    if c_label_index_len > batch_each_class:
                        add_index_vector = np.random.choice(c_label_index_len, batch_each_class, replace=False)
                        if (i<3):
                            logger.info("add index vector for c " + str(c_label))
                            logger.info(add_index_vector)
                        add_index_vector_len = len(add_index_vector)
                        batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                        batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                    else:
                        batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                        batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                        add_index_vector_len = c_label_index_len
                else:
                    batch_class_index = batch_class_index_dict[c_label]
                    add_index_vector_len = len(batch_class_index)
                    c_label_index = train_class_index_dict[c_label]
                    c_label_index_len = len(c_label_index)
                    if add_index_vector_len < batch_each_class:
                        add_count = batch_each_class - add_index_vector_len
                        if c_label_index_len > add_count:
                            add_index_vector = np.random.choice(c_label_index_len, add_count, replace=False)
                            if (i<3):
                                logger.info("add index vector for c " + str(c_label))
                                logger.info(add_index_vector)
                            add_index_vector_len = add_index_vector_len + len(add_index_vector)
                            batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                            batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                        else:
                            batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                            batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                            add_index_vector_len = add_index_vector_len + c_label_index_len
                    elif add_index_vector_len > 2 * batch_each_class:
                        remove_count = (add_index_vector_len - 2 * batch_each_class)
                        remove_index_vector = np.random.choice(batch_class_index, remove_count, replace=False)
                        add_index_vector_len = add_index_vector_len - len(remove_index_vector)
                        batch_x_matrix = np.delete(batch_x_matrix, remove_index_vector, axis=0)
                        batch_y_matrix = np.delete(batch_y_matrix, remove_index_vector, axis=0)
                        batch_y_vector = np.argmax(batch_y_matrix, axis=1)
                        batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
                coefficients_vector.append(float(add_index_vector_len))
            #print "End of F1"
            
                
            coefficients_vector = np.array(coefficients_vector)
            batch_max_len = float(max(coefficients_vector))
            coefficients_vector = batch_max_len/coefficients_vector
            if i < 3:
                batch_y_vector = np.argmax(batch_y_matrix, axis=1)
                batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
                logger.info("class index after: ")
                logger.info(batch_class_index_dict)
                logger.info("coefficient vector: ")
                logger.info(coefficients_vector)
            # End of BATCH_CONTROLLED
            #print "shape 2"
            #print batch_x_matrix.shape
            #print batch_y_matrix.shape
            #print "===="

            start_time = time.time()
            train_step.run(feed_dict={input_x_placeholder: batch_x_matrix,
                                  output_y_placeholder: batch_y_matrix, coefficient_placeholder:coefficients_vector, keep_prob: 1})
            train_run_time = train_run_time + time.time() - start_time
        else:
            start_time = time.time()
            train_step.run(feed_dict={input_x_placeholder: batch_x_matrix,
                                  output_y_placeholder: batch_y_matrix, keep_prob: 1})
            train_run_time = train_run_time + time.time() - start_time
        if i % 100 == 0:
            fir_weight_variable = tf.get_default_graph().get_tensor_by_name("conv_w_0:0")
            logger.info("fir weight")
            logger.info(fir_weight_variable.get_shape())
            fir_weight_var_val = cnn_session.run(fir_weight_variable)
            logger.info(fir_weight_var_val[0, 0:5, 0, 0])
            test_eval_value = eval_method_value.eval(feed_dict={
                input_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob: 1})
            if str(test_eval_value) == 'nan':
                test_eval_value = 0
            print_str = "step " + str(i) + ", testing " + eval_method_keyword + ": " + str(test_eval_value)
            logger.info(print_str)
            if best_eval_value < test_eval_value:
                # Save the variables to disk.
                best_eval_value = test_eval_value
                save_path = saver.save(cnn_session, saver_file)
                print_str = "Model saved in file: " + save_path + ' at iteration: ' + str(i)
                logger.info(print_str)
        
        i = i + 1
        start = end
        end = end + batch_size
        if epoch > max_iter:
            logger.info("best eval value at epoch: " + str(epoch))
            logger.info("best eval value to break")
            logger.info(best_eval_value)
            break
            #if str(best_eval_value) == 'nan' or best_eval_value < 0.1:
            #    if re_init == False:
            #        epoch = 0 
            #        logger.info("re initialize the variables")
            #        cnn_session.run(tf.global_variables_initializer())
            #        batch_index = np.random.permutation(overall_len)
            #        start = 0
            #        end = batch_size
            #        re_init = True
            #    else:
            #        break
            #else:
            #    break
        


    start_time = time.time()
    test_eval_value = eval_method_value.eval(feed_dict={
                                       input_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob: 1})
    test_run_time = time.time() - start_time
    if test_eval_value < best_eval_value:
        cnn_session.close()
        cnn_session = tf.InteractiveSession()
        saver.restore(cnn_session, saver_file)
    else:
        best_eval_value = test_eval_value
    
    #if best_eval_value == 0:
    #    return 
    logger.info("Running iteration: %d" % (i))
    logger.info("final best " + eval_method_keyword + ": " + str(best_eval_value))
    logger.info(f1_unbalance_count)


    cnn_predict_proba = cnn_session.run(predict_y_proba, feed_dict={input_x_placeholder: test_x_matrix, keep_prob: 1.0})
    logger.info("CNN model saved: " + str(saver_file))
    
    if cnn_setting.feature_method == 'none':
        cnn_session.close()
        return best_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, ''
    
    #keeped_feature_value_list = []
    logger.info("feature value generation")
    #for feature_placeholder in keeped_feature_list:
    #    feature_value = feature_placeholder.eval(feed_dict={input_x_placeholder: train_x_matrix, keep_prob: 1.0})
    #    keeped_feature_value_list.append(feature_value)
    #    logger.info(feature_value.shape)
    test_keeped_feature_value_list = cnn_session.run(keeped_feature_list, feed_dict={input_x_placeholder: test_x_matrix, keep_prob: 1.0})
    logger.info('test feature list ready')
    start = 0
    end = 0
    train_row = len(train_x_matrix)
    train_obj_list = []
    while(start<train_row):
        logger.info(start)
        end = start + 1000
        if end > train_row:
            end = train_row
        keep_obj = cnn_session.run(keeped_feature_list[0], feed_dict={input_x_placeholder: train_x_matrix[start:end, :, :, :], keep_prob: 1.0})
        train_obj_list.append(keep_obj)
        start = end
    #keeped_feature_value_list = cnn_session.run(keeped_feature_list, feed_dict={input_x_placeholder: train_x_matrix, keep_prob: 1.0})
    logger.info('train feature list ready')
    logger.info("The order of feature value list: fir_out_conv_no_act, fir_out_conv, fir_weight, fir_bias, last_conv, weight_full, bias_full")
    logger.info("All features saved to ")
    logger.info("CNN feature list saved to: " + feature_obj_file)
    save_obj([train_obj_list, test_keeped_feature_value_list], feature_obj_file)
    cnn_session.close()
    return best_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_obj_file



          
def ori_cnn_train(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, input_x_placeholder, output_y_placeholder, keep_prob, predict_y_proba, feature_result, weight_fullconn, bias_fullconn, saver_file, logger=None):
    if logger == None:
        logger = init_logging('')
    min_class = 0
    eval_method = cnn_setting.eval_method
    batch_size = cnn_setting.batch_size
    stop_threshold = cnn_setting.stop_threshold
    max_iter = cnn_setting.max_iter
    feature_method = cnn_setting.feature_method
    
    prediction = tf.argmax(predict_y_proba, 1)
    actual = tf.argmax(output_y_placeholder, 1)

    correct_prediction = tf.equal(prediction, actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if eval_method=='f1':
        train_y_vector = np.argmax(train_y_matrix, axis=1)
        train_class_index_dict, train_min_length, train_max_length = class_label_vector_checking(train_y_vector)
        min_class = 0
        max_class = max(train_y_vector)
        num_classes = max_class + 1
        if max_class == 1:
            TP = tf.count_nonzero(prediction * actual, dtype=tf.float32)
            TN = tf.count_nonzero((prediction - 1) * (actual - 1), dtype=tf.float32)
            FP = tf.count_nonzero(prediction * (actual - 1), dtype=tf.float32)
            FN = tf.count_nonzero((prediction - 1) * actual, dtype=tf.float32)
            precision = (TP) / (TP + FP)
            recall = (TP) / (TP + FN)
            f1 = (2 * precision * recall) / (precision + recall)
            eval_method_value = f1
        coefficient_placeholder = tf.placeholder(tf.float32, shape=[num_classes])
        cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=output_y_placeholder, logits=predict_y_proba, pos_weight=coefficient_placeholder))
    #elif eval_method=='mean_acc':
    #    train_y_vector = np.argmax(train_y_matrix, axis=1)
    else:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_y_placeholder, logits=predict_y_proba))
        eval_method_value = accuracy

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    cnn_session = tf.InteractiveSession()
    cnn_session.run(tf.global_variables_initializer())

    test_eval_value = -1
    best_eval_value = -1
    i = 0
    start = 0
    epoch = 0
    end = batch_size
    batch_each_class = int(batch_size/num_classes)
    overall_len = len(train_y_matrix)

    saver = tf.train.Saver()
    train_run_time = 0
    batch_index = np.random.permutation(overall_len)
    f1_unbalance_count = np.zeros(num_classes)

    while(test_eval_value < stop_threshold):
        if start >= overall_len:
            start = 0
            end = start + batch_size
            epoch = epoch + 1
            batch_index = np.random.permutation(overall_len)
        elif end > overall_len:
            end = overall_len
        batch_x_matrix = train_x_matrix[batch_index[start:end], :, :, :]
        batch_y_matrix = train_y_matrix[batch_index[start:end], :]

        if eval_method == 'f1':
            ### Normal BATCH Weight
            #batch_y_vector = np.argmax(batch_y_matrix, axis=1)
            #batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
            #coefficients_vector = []
            #batch_class_index_dict_keys = batch_class_index_dict.keys()
            #for c_label in range(min_class, max_class+1):
            #    if c_label not in batch_class_index_dict_keys:
            #        add_index_vector_len = 0.1
            #    else:
            #        add_index_vector_len = len(batch_class_index_dict[c_label])
            #    coefficients_vector.append(float(batch_max_length)/float(add_index_vector_len))
            #coefficients_vector = np.array(coefficients_vector)
            ### End of Normal BATCH Weight

            # BATCH_CONTROLED
            batch_y_vector = np.argmax(batch_y_matrix, axis=1)
            batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
            coefficients_vector = []
            batch_class_index_dict_keys = batch_class_index_dict.keys()
            for c_label in range(min_class, max_class+1):
                #print "class: " + str(c_label)
                #print class_label_vector_checking
                if c_label not in batch_class_index_dict_keys:
                    f1_unbalance_count[c_label] = f1_unbalance_count[c_label] + 1
                    c_label_index = train_class_index_dict[c_label]
                    c_label_index_len = len(c_label_index)
                    add_index_vector_len = 0
                    if c_label_index_len > batch_max_length:
                        add_index_vector = np.random.choice(c_label_index_len, batch_max_length, replace=False)
                        add_index_vector_len = len(add_index_vector)
                        batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                        batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                    else:
                        batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                        batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                        add_index_vector_len = c_label_index_len
                else:
                    add_index_vector_len = len(batch_class_index_dict[c_label])
                    c_label_index = train_class_index_dict[c_label]
                    c_label_index_len = len(c_label_index)
                    if add_index_vector_len < batch_each_class:
                        add_count = batch_each_class - add_index_vector_len
                        if c_label_index_len > add_count:
                            add_index_vector = np.random.choice(c_label_index_len, add_count, replace=False)
                            add_index_vector_len = add_index_vector_len + len(add_index_vector)
                            batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                            batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                        else:
                            batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                            batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                            add_index_vector_len = add_index_vector_len + c_label_index_len
                coefficients_vector.append(float(batch_max_length)/float(add_index_vector_len))
            #print "End of F1"
            coefficients_vector = np.array(coefficients_vector)
            # End of BATCH_CONTROLED
            

            start_time = time.time()
            train_step.run(feed_dict={input_x_placeholder: batch_x_matrix,
                                  output_y_placeholder: batch_y_matrix, coefficient_placeholder:coefficients_vector, keep_prob: 1})
            train_run_time = train_run_time + time.time() - start_time
        else:
            start_time = time.time()
            train_step.run(feed_dict={input_x_placeholder: batch_x_matrix,
                                  output_y_placeholder: batch_y_matrix, keep_prob: 1})
            train_run_time = train_run_time + time.time() - start_time

        if i % 100 == 0:
            test_eval_value = eval_method_value.eval(feed_dict={
                input_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob: 1})
            if str(test_eval_value) == 'nan':
                test_eval_value = 0
            print_str = "step " + str(i) + ", testing " + eval_method + ": " + str(test_eval_value)
            logger.info(print_str)
            if best_eval_value < test_eval_value:
                # Save the variables to disk.
                best_eval_value = test_eval_value
                save_path = saver.save(cnn_session, saver_file)
                print_str = "Model saved in file: " + save_path + ' at iteration: ' + str(i)
                logger.info(print_str)
        
        i = i + 1
        if epoch > max_iter:
            if str(best_eval_value) == 'nan' or best_eval_value < 0.1:
                epoch = 0 
                cnn_session.run(tf.global_variables_initializer())
                batch_index = np.random.permutation(overall_len)
            else:
                break
        start = end
        end = end + batch_size

    start_time = time.time()
    test_eval_value = eval_method_value.eval(feed_dict={
                                       input_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob: 1})
    test_run_time = time.time() - start_time
    if test_eval_value < best_eval_value:
        cnn_session.close()
        cnn_session = tf.InteractiveSession()

        saver.restore(cnn_session, saver_file)       
        #test_eval_value = eval_method_value.eval(feed_dict={input_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob: 1.0})
    else:
        best_eval_value = test_eval_value

    if str(best_eval_value) == 'nan':
        best_eval_value = 0
    logger.info("Running iteration: %d" % (i))
    logger.info("final best " + eval_method + ": " + str(best_eval_value))
    #logger.info("final test " + eval_method + ": " + str(test_eval_value))
    logger.info(f1_unbalance_count)
    
    cnn_predict_proba = cnn_session.run(predict_y_proba, feed_dict={input_x_placeholder: test_x_matrix, keep_prob: 1.0})
    logger.info("CNN model saved: " + str(saver_file))
    if feature_method == 'none':
        return best_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file

    #train_feature_result = cnn_session.run(feature_result, feed_dict={input_x_placeholder: test_x_matrix, keep_prob: 1.0})
    train_feature_result = feature_result.eval(feed_dict={input_x_placeholder: train_x_matrix, keep_prob: 1.0})
    #print input_x_placeholder.get_shape()
    #print train_x_matrix.shape
    #train_feature_result = cnn_session.run(feature_result, feed_dict={
    #                                       input_x_placeholder: train_x_matrix, keep_prob: 1.0})
    #train_feature_result = feature_result.eval(feed_dict={input_x_placeholder: train_x_matrix, keep_prob: 1.0})
    logger.info(train_feature_result.shape)

    #weight_fullconn = cnn_session.run(weight_fullconn)
    #full_weight_row, full_weight_col = weight_fullconn.shape
    #logger.info('full_weight')
    #logger.info(weight_fullconn.shape)

    #bias_fullconn = cnn_session.run(bias_fullconn)
    #full_bias_len = bias_fullconn.shape
    #logger.info('full_bias')
    #logger.info(full_bias_len)
    logger.info("CNN last conv layer saved: " + str(saver_file+"_last_conv_layer_output.pckl"))
    train_feature_result = np.squeeze(train_feature_result)
    save_obj([train_feature_result], saver_file + "_last_conv_layer_output.pckl")
    #save_obj([train_feature_result, weight_fullconn, bias_fullconn], saver_file+"_last_conv_layer_output.pckl")
    cnn_session.close()
    return best_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file


## Input matrix: <input_row, attr_len, attr_num, input_count>
#def conf_conv_no_sum_layer(layer, kernel_r, input_matrix, num_input_map, num_output_map, group_list=[], activation_fun=0, stride_list=[1,1,1,1], std_value=0.002, same_size=False):
#    std_value = sqrt(0.2 / num_input_map)
#    input_row, attr_len, input_count, attr_num = input_matrix.get_shape()
#    attr_num = int(attr_num)
#    last_conv = False
#    if kernel_r <= 0:
#        last_conv = True
#        kernel_r = int(attr_len)
#    weight_variable = tf.Variable(tf.truncated_normal([kernel_r, 1, attr_num, num_output_map], stddev=std_value), name='conv_weight_'+str(layer))
#    bias_variable = tf.Variable(tf.constant(0.0, shape=[num_output_map]), name='conv_bias_'+str(layer))
#
#    if same_size == "True":
#        str_padding = 'SAME'
#    else:
#        str_padding = 'VALID'
#    
#    output_conv = conv_updated(input_matrix, weight_variable, stride_list, str_padding)
#    if len(group_list) > 0:
#        output_conv = conv_no_sum_grouping(output_conv, group_list) + bias_variable
#    if activation_fun == 0:
#        ret_conv = tf.nn.relu(output_conv)
#    elif activation_fun == 1:
#        ret_conv = tf.nn.sigmoid(output_conv)
#    elif activation_fun == 2:
#        ret_conv = tf.nn.tanh(output_conv)
#    elif activation_fun == -1:
#        ret_conv = output_conv
#    if last_conv == False:
#        ret_conv = tf.reduce_sum(ret_conv, 4)
#    else:
#        fir_row, sec_row, thi_row, for_row, fif_row = ret_conv.get_shape()
#        ret_conv = tf.reshape(ret_conv, [-1, int(sec_row), int(for_row), int(fif_row)])
#        
#    return ret_conv, weight_variable, bias_variable


def conf_pool_layer(input_matrix, row_d_samp_rate, col_samp_rate, same_size=False):
    if same_size == True:
        str_padding = 'SAME'
    else:
        str_padding = 'VALID'
    return tf.nn.max_pool(input_matrix, ksize=[1, row_d_samp_rate, col_samp_rate, 1], strides=[1, row_d_samp_rate, col_samp_rate, 1], padding=str_padding)
    

def conf_out_layer(input_x_matrix, num_features, num_classes, std_value=0.1):
    #std_value = sqrt(0.2 / num_features)
    #std_value = 0.1
    tf.random.set_random_seed(0)
    output_weight = tf.Variable(tf.truncated_normal(
        [num_features, num_classes], stddev=std_value))
    output_bias = tf.Variable(tf.constant(std_value, shape=[num_classes]))
    predict_y_proba = tf.matmul(input_x_matrix, output_weight) + output_bias
    predict_y_proba = tf.nn.softmax(predict_y_proba)
    return predict_y_proba#, output_weight, output_bias


#def conf_softmax(input_matrix, num_input_map, num_output_map):
#    W = tf.Variable(tf.zeros([num_input_map, num_output_map]))
#    b = tf.Variable(tf.zeros([num_output_map]))
#    return tf.nn.softmax(tf.matmul(input_matrix, W) + b)


# grouped in the first layer or not
# train_x_placeholder: row, attr_len, attr_num, input_map
# group_all: True means group all attributes in the first layer
def cnn_configure(train_x_placeholder, cnn_setting, num_classes, group_all=False, logger=None):
    if logger == None:
        logger = init_logging('')

    # CNN Parameters
    conv_kernel_list = cnn_setting.conv_kernel_list
    pool_rate_list = cnn_setting.pool_rate_list
    feature_num_list = cnn_setting.feature_num_list
    activation_fun = cnn_setting.activation_fun
    std_value = cnn_setting.std_value
    same_size = cnn_setting.same_size
    cnn_group_list = cnn_setting.group_list
    conv_row_num = len(conv_kernel_list)
    saver_file = ''

    keeped_feature_list = []

    num_input_map = cnn_setting.input_map
    strides_list = [1, 1, 1, 1]


    relu_base_array = []
    for i in range(0, conv_row_num + 1):
        relu_base_array.append(None)

    for i in range(0, conv_row_num):
        logger.info('layer: ' + str(i) + " input:")
        logger.info(train_x_placeholder.get_shape())
        conv_row_kernel = conv_kernel_list[i, 0]
        conv_col_kernel = conv_kernel_list[i, 1]
            
        train_x_row = int(train_x_placeholder.get_shape()[1])
        train_x_col = int(train_x_placeholder.get_shape()[2])

        if conv_row_kernel < 0:
            conv_row_kernel = train_x_row
        elif conv_row_kernel > train_x_row:
            conv_row_kernel = train_x_row

        num_output_map = feature_num_list[i]
        if i == 0 and group_all==True:
            conv_col_kernel = train_x_col
        elif conv_col_kernel > train_x_col:
            conv_col_kernel = train_x_col
        elif conv_col_kernel < 0:
            conv_col_kernel = train_x_col

        saver_file = saver_file + "_c" + str(conv_row_kernel) + "_" + str(conv_col_kernel)
        #activation_fun = 3
        #print i, conv_row_kernel, conv_col_kernel, train_x_placeholder, num_input_map, num_output_map, activation_fun, strides_list, std_value, same_size
        out_conv, relu_base = conf_conv_layer(i, conv_row_kernel, conv_col_kernel, train_x_placeholder, num_input_map, num_output_map, activation_fun, strides_list, std_value, same_size, relu_base_array[i])
        relu_base_array[i] = relu_base
        logger.info("Conv output: " + str(out_conv.get_shape()))
        pool_row_kernel = pool_rate_list[i, 0]
        pool_col_kernel = pool_rate_list[i, 1]

        saver_file = saver_file + "_p" + str(pool_row_kernel) + "_" + str(pool_col_kernel)

        out_conv_row = int(out_conv.get_shape()[1])
        out_conv_col = int(out_conv.get_shape()[2])

        if pool_row_kernel > 0 and pool_col_kernel > 0:
            if pool_row_kernel > out_conv_row:
                warning_str = "Warning: given pooling row number " + str(pool_row_kernel) + \
                    " is bigger than the data row number " + str(out_conv_row)
                logger.info(warning_str)
                warning_str = "Setting the pooling row number to be the data row number"
                logger.info(warning_str)
                pool_row_kernel = out_conv_row
            if pool_col_kernel > out_conv_col:
                warning_str = "Warning: given pooling column number " + \
                    str(pool_col_kernel) + \
                    " is bigger than the data column number " + \
                    str(out_conv_row)
                logger.info(warning_str)
                warning_str = "Setting the pooling column number to be the data column number"
                logger.info(warning_str)
                pool_col_kernel = out_conv_col
            train_x_placeholder = conf_pool_layer(out_conv, pool_row_kernel, pool_col_kernel, same_size)
            logger.info("Pooling output: " + str(train_x_placeholder.get_shape()))
        else:
            train_x_placeholder = out_conv
        num_input_map = num_output_map

    saver_file = saver_file + '.ckpt'
    #############################################
    # typical full connect layer
    # print final_conv_kernel
    last_out_conv = train_x_placeholder
    # Only save the matrix before fully connected layer
    keeped_feature_list.append(last_out_conv)
    logger.info("Feature result shape")
    logger.info(last_out_conv.get_shape())
    #print "last out conv"
    #print last_out_conv.get_shape()
    second_feature_num = int(last_out_conv.get_shape()[1] * last_out_conv.get_shape()[2] * last_out_conv.get_shape()[3])
    output_feature_num = 400

    #std_value = sqrt(2.0 / second_feature_num)
    #std_value = 0.02
    tf.random.set_random_seed(0)
    weight_fullconn = tf.Variable(tf.truncated_normal([second_feature_num, output_feature_num], stddev=std_value))
    logger.info("full conn weight shape")
    logger.info(weight_fullconn.get_shape())
    bias_fullconn = tf.Variable(tf.constant(std_value, shape=[output_feature_num]))
    #keeped_feature_list.append(weight_fullconn)
    #keeped_feature_list.append(bias_fullconn)
    h_pool2_flat = tf.reshape(last_out_conv, [-1, second_feature_num])
    output_fullconn_no_act = tf.matmul(h_pool2_flat, weight_fullconn) + bias_fullconn

    output_fullconn, relu_base_array = conf_act(output_fullconn_no_act, activation_fun, relu_base_array[-1])

    logger.info('last full connect layer output:')
    logger.info(str(output_fullconn.get_shape()))
    
    #dropout
    keep_prob_placeholder = tf.placeholder(tf.float32)
    output_fullconn_drop = tf.nn.dropout(output_fullconn, keep_prob_placeholder)

    #predict_y_proba = confSoftmax(output_fullconn_drop, output_feature_num, num_classes)
    #print output_fullconn_drop.get_shape()
    #print output_feature_num
    #print num_classes
    #print std_value
    predict_y_proba = conf_out_layer(output_fullconn_drop, output_feature_num, num_classes, std_value)
    #print "predict_y_proba"
    #print predict_y_proba.get_shape()
    return predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, relu_base_array


# End of CNN method




## CNN load and predict
def load_model(model_saved_file, data_stru, cnn_setting, group_all=True, logger=None):
    if logger == None:
        logger = init_logging('')
    logger.info("load model function")
    logger.info(data_stru.attr_num)
    logger.info(data_stru.attr_len)
    input_map = cnn_setting.input_map
    logger.info(cnn_setting.to_string())

    train_x_placeholder, output_y_placeholder, predict_y_proba, keep_prob_placeholder, keeped_feature_list, saver_file, relu_base_array = cnn_set_flow_graph(data_stru, cnn_setting, input_map, group_all, logger)

    cnn_session = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(cnn_session, model_saved_file)
    return cnn_session, predict_y_proba, train_x_placeholder, keep_prob_placeholder



def load_model_predict(cnn_session, test_x_matrix, predict_y_proba,train_x_placeholder, keep_prob_placeholder):
    cnn_predict_proba = cnn_session.run(predict_y_proba, feed_dict={
                                        train_x_placeholder: test_x_matrix, keep_prob_placeholder: 1.0})
    return cnn_predict_proba

## End of CNN load and predict



def activation_rf(feature_tensor, y_vector, num_classes, num_trees=50, max_nodes=1000):
    from tensorflow.contrib.tensor_forest.python import tensor_forest
    from tensorflow.python.ops import resources
    train_row, attr_len, attr_num, feature_map = feature_tensor.shape
    num_features = attr_len * attr_num * feature_map
    feature_tensor = tf.reshape(feature_tensor, [train_row, num_features])
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
    


def activation_dy_relu(feature_tensor, attr_threshold_variable):
    # feature_num is the number of features learned in this layer
    train_row, feature_num, attr_num, feature_map = feature_tensor.shape
    feature_value = feature_tensor.eval()
    #add_threshold_place = tf.zeros([train_row, attr_len, attr_num, feature_map], tf.float32)

    #for attr_id in range(attr_num):
    #    add_threshold_place[:, attr_id, :, :] = add_threshold_place[:, attr_id, :, :] + attr_threshold_variable[attr_id]
        #feature_tensor[:, :, attr_id, :] = feature_tensor[:, :, attr_id, :] - attr_threshold_variable[attr_id]
    feature_tensor = feature_tensor - attr_threshold_variable
    return tf.nn.relu(feature_tensor)
    #return feature_tensor



if __name__ == '__main__':

    input_row = 3
    attr_num = 4
    attr_len = 6
    kernel_r = 1
    kernel_c = 3
    input_count = 1
    output_count = 1

    session = tf.InteractiveSession()
    I_value = np.random.randint(5, size=(input_row, attr_len, input_count, attr_num))
    I_value = I_value - 3
    I_placeholder = tf.placeholder(tf.float32, [None, attr_len, input_count, attr_num])
    
    #attr_threshold_value = [2, 1, 0, -1]
    attr_threshold_variable = tf.Variable(tf.truncated_normal([attr_num], stddev=0.1))
        

    relu_out = tf.nn.relu(I_placeholder)
    dy_relu_out = activation_dy_relu(I_placeholder, attr_threshold_variable)
    session.run(tf.global_variables_initializer())
    relu_out_value = relu_out.eval(feed_dict={I_placeholder:I_value})
    print I_value
    #print relu_out_value
    dy_relu_out_value = dy_relu_out.eval(feed_dict={I_placeholder:I_value})
    #print I_value
    attr_threshold_value = session.run(attr_threshold_variable)
    print dy_relu_out_value
    print attr_threshold_value


    sdfs

    
    #I_value = I_value.reshape((1, attr_num, attr_len, input_count))
    print "TEST"
    print I_value[0, :, :, 0]
    #I_value = np.transpose(I_value, (0, 1, 2, 1))
    print "after"
    print I_value[0, :, :, 0]
    print I_value.shape

    print "Start"

    w_value = np.ones((kernel_c, kernel_r, attr_num, output_count))

    I_placeholder = tf.placeholder(tf.float32, [None, attr_len, input_count, attr_num])
    #W_placeholder = tf.placeholder(tf.float32, [kernel_c, kernel_r, input_count, output_count])
    W_placeholder = tf.placeholder(tf.float32, [kernel_c, input_count, attr_num, output_count])
    print I_placeholder.get_shape()
    
    #print W_placeholder.get_shape()

    up_output = conv_updated(I_placeholder, W_placeholder)
    up_fir, up_sec, up_thi, up_for = up_output.get_shape()
    up_output = tf.reshape(up_output, [-1, up_sec, output_count, attr_num])
    print "up_output.get_shape()"
    print up_output.get_shape()

    W2_placeholder = tf.placeholder(tf.float32, [kernel_c, output_count, attr_num, 3])
    w2_value = np.ones((kernel_c, output_count, attr_num, 3), dtype=int)
    up_output_2 = conv_updated(up_output, W2_placeholder)
    up_fir, up_sec, up_thi, up_for = up_output_2.get_shape()
    up_output_2 = tf.reshape(up_output_2, [-1, up_sec, 3, attr_num])
    print "up_output_2.get_shape()"
    print up_output_2.get_shape()
    group_list = [1, 2, 3, 2, 1]
    final_conv_output = conv_no_sum_grouping(up_output_2, group_list)
    
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    output_value = up_output.eval(feed_dict={I_placeholder:I_value, W_placeholder:w_value})
    up_output_2_value = up_output_2.eval(feed_dict={I_placeholder:I_value, W_placeholder:w_value, W2_placeholder:w2_value})
    final_conv_output_value = final_conv_output.eval(feed_dict={I_placeholder:I_value, W_placeholder:w_value, W2_placeholder:w2_value})
    print "I value"
    print I_value[0, :, 0, :]
    print I_value.shape
    print "w value"
    print w_value[:, :, 0, 0]
    print "out value"
    print output_value[0, :, 0, :]
    print output_value.shape
    print "out 2 value"
    print up_output_2_value[0, :, 0, :]
    print up_output_2_value[0, :, 0, :]
    print up_output_2.shape

    print "final out conv"
    #print final_conv_output_value
    print final_conv_output_value.shape
    print group_list

    
    

    #sdf
#
    ##I_value = I_value.reshape(-1, 1, attr_len, attr_num)
    ##w_value = w_value.reshape(1, kernel_c, kernel_r, 1)
    #I = tf.reshape(I_placeholder, [-1, input_count, attr_len, attr_num])
    #W = tf.reshape(W_placeholder, [input_count, kernel_c, kernel_r, output_count])
    #strides = [1,1,1,1]
    #output = tf.nn.depthwise_conv2d(I, W, strides, padding='VALID')
   #
    #session = tf.InteractiveSession()
    #session.run(tf.global_variables_initializer())
    #output_value = output.eval(feed_dict={I_placeholder:I_value, W_placeholder:w_value})
    #print "I"
    #I_value = I_value.reshape(-1, attr_num, attr_len, input_count)
    #print I_value[0, :, :, 0]
    #print I_value.shape
    #print "W"
    #print w_value
    #print "out"
    ##output_value = output_value.reshape(attr_num, attr_len)
    #print output_value
    #print output_value.shape
    #output_value = output_value.reshape(-1, 1, attr_len-kernel_c+1, attr_num, output_count)
    #print output_value.shape
    #print output_value[0, 0, :, 0, 0]
    #print output_value[0, 0, :, 1, 0]
    #print output_value[0, 0, :, 1, 0]
    
    # #ret_conv.eval(feed_dict={train_x_placeholder: train_x_matrix})


   
   
    # layer = 0
    # kernel_r = 1
    # kernel_c = 2

    # train_x_matrix = np.random.rand(1, 15)
    # train_x_matrix = train_x_matrix.reshape(1, 5, 3, 1)
    # print train_x_matrix[0, :, 0, 0]
    # attr_num = 5
    # attr_len = 3
   
    # train_col_num = 10
    # input_x_placeholder = tf.placeholder(tf.float32, [None, train_col_num])
    # train_x_placeholder = tf.reshape(input_x_placeholder, [-1, attr_num, attr_len, 1])
    # num_input_map = 1
    # num_output_map = 1
    # activation_fun = 0
    # std_value = 0.002
    # same_size = False
    # #ret_conv, weight_variable, bias_variable = confConvLayer(layer, kernel_r, kernel_c, train_x_placeholder, num_input_map, num_output_map, activation_fun, std_value, same_size)
    
    # #cnn_session = tf.InteractiveSession()
    # #cnn_session.run(tf.global_variables_initializer())
    # #ret_conv.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # #print ret_conv.shape

    # kernel_r_dict = {}
    # kernel_r_dict[0] = {0, 1}
    # kernel_r_dict[1] = {2, 3}
    # kernel_r_dict[2] = {4}
    # ret_conv_1, weight_variable_1, bias_variable_1, fir_input_matrix, sec_input_matrix, thi_input_matrix = conf_varying_conv_layer(layer, kernel_r_dict, kernel_c, train_x_placeholder, num_input_map, num_output_map, activation_fun, std_value, same_size)
    # #ret_conv_1 = conf_varying_conv_layer(layer, kernel_r_dict, kernel_c, train_x_placeholder, num_input_map, num_output_map, activation_fun, std_value, same_size)
    # cnn_session = tf.InteractiveSession()
    # cnn_session.run(tf.global_variables_initializer())
    
    # print train_x_matrix

    # ret_conv_1_value = ret_conv_1.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # print ret_conv_1_value.shape
    # print ret_conv_1_value
    
    
    # weight_variable_1_value = weight_variable_1.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # print weight_variable_1_value.shape
    # bias_variable_1_value = bias_variable_1.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # print bias_variable_1_value.shape


    # print train_x_matrix[0, :, :, 0]
    # print "======"
    # fir_input_matrix_value = fir_input_matrix.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # print fir_input_matrix_value[0, :, :, 0]
    # print fir_input_matrix_value.shape
    # sec_input_matrix_value = sec_input_matrix.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # print sec_input_matrix_value[0, :, :, 0]
    # print sec_input_matrix_value.shape
    # thi_input_matrix_value = thi_input_matrix.eval(feed_dict={train_x_placeholder: train_x_matrix})
    # print thi_input_matrix_value[0, :, :, 0]
    # print thi_input_matrix_value.shape


    # #print ret_conv_1_value
    # #print train_x_matrix[0, 0, :, 0]
    # #print train_x_matrix[0, 1, :, 0]
    
