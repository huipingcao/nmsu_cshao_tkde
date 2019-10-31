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


def conf_conv_layer(layer, kernel_r, kernel_c, input_matrix, num_input_map, num_output_map, activation_fun=0, strides_list=[1,1,1,1], std_value=0.1, same_size=False, logger=None):
    #if layer == 0:
    #    std_value = sqrt(0.2)
    #else:
    #    std_value = sqrt(0.2 / num_input_map)
    if logger is None:
        logger = init_logging("")
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

    ret_conv = tf.nn.relu(ret_conv_before_act)
    return ret_conv


def cnn_set_flow_graph(data_stru, cnn_setting, input_map, group_all=False, logger=None):
    if logger is None:
        logger = init_logging('')
    tf.reset_default_graph()
    tf.random.set_random_seed(0)

    attr_num = data_stru.attr_num
    attr_len = data_stru.attr_len
    num_classes = data_stru.num_classes

    output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
    train_x_placeholder = tf.placeholder(tf.float32, [None, attr_len, attr_num, input_map])
    logits_out, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_configure(train_x_placeholder, cnn_setting, num_classes, group_all, logger)
    return train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file



############################################################################################################################################


#train_x_matrix: train_row, attr_len, attr_num, input_map
#test_x_matrix: test_row, attr_len, attr_num, input_map
def run_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, group_all=False, saver_file_profix='', logger=None):
    if logger is None:
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

    train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_set_flow_graph(data_stru, cnn_setting, input_map, group_all, logger)

    saver_file = saver_file_profix + "_group_" + str(group_all) + saver_file
    cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file = cnn_train(
        train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
    if str(cnn_eval_value) == 'nan':
        cnn_eval_value = 0
    return cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file
    #return cnn_accuracy, train_run_time, test_run_time, cnn_predict_proba, train_sensor_result, weight_fullconn, bias_fullconn


#train_x_matrix: train_row, attr_len, attr_num, input_map
#test_x_matrix: test_row, attr_len, attr_num, input_map
def run_projected_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, group_all=False, saver_file_profix='', logger=None):
    if logger is None:
        logger = init_logging('')
    num_classes = data_stru.num_classes
    attr_num = data_stru.attr_num
    attr_len = data_stru.attr_len
    logger.info(cnn_setting)

    train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    data_stru.attr_num = attr_num
    data_stru.attr_len = attr_len

    train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_set_flow_graph(
        data_stru, cnn_setting, input_map, group_all, logger)

    saver_file = saver_file_profix + "_group_" + str(group_all) + saver_file
    cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file = cnn_train(
        train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
    if str(cnn_eval_value) == 'nan':
        cnn_eval_value = 0
    return cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file
    #return cnn_accuracy, train_run_time, test_run_time, cnn_predict_proba, train_sensor_result, weight_fullconn, bias_fullconn



def cnn_train(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, num_classes, cnn_setting, input_x_placeholder, output_y_placeholder, logits_out, keep_prob, keeped_feature_list, saver_file="./", logger=None):
    if logger is None:
        logger = init_logging('')
    min_class = 0
    eval_method = cnn_setting.eval_method
    batch_size = cnn_setting.batch_size
    stop_threshold = cnn_setting.stop_threshold
    max_iter = cnn_setting.max_iter
    feature_method = cnn_setting.feature_method
    feature_obj_file = cnn_setting.out_obj_folder + saver_file
    saver_file = cnn_setting.out_model_folder + saver_file
    predict_y_proba = tf.nn.softmax(logits_out)
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
            targets=output_y_placeholder, logits=logits_out, pos_weight=coefficient_placeholder))
    else:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_y_placeholder, logits=logits_out))
        eval_method_value = accuracy
        eval_method_keyword = "acc"
    #print cross_entropy.get_shape()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
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
    logits_out = tf.matmul(input_x_matrix, output_weight) + output_bias
    return logits_out#, output_weight, output_bias


# grouped in the first layer or not
# train_x_placeholder: row, attr_len, attr_num, input_map
# group_all: True means group all attributes in the first layer
def cnn_configure(train_x_placeholder, cnn_setting, num_classes, group_all=False, logger=None):
    if logger is None:
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
        out_conv = conf_conv_layer(i, conv_row_kernel, conv_col_kernel, train_x_placeholder, num_input_map, num_output_map, activation_fun, strides_list, std_value, same_size, logger)
        
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

    output_fullconn = tf.nn.relu(output_fullconn_no_act)
    logger.info('last full connect layer output:')
    logger.info(str(output_fullconn.get_shape()))
    
    #dropout
    keep_prob_placeholder = tf.placeholder(tf.float32)
    output_fullconn_drop = tf.nn.dropout(output_fullconn, keep_prob_placeholder)

    logits_out = conf_out_layer(output_fullconn_drop, output_feature_num, num_classes, std_value)
    #print "logits_out"
    #print logits_out.get_shape()
    return logits_out, keep_prob_placeholder, keeped_feature_list, saver_file


# End of CNN method
