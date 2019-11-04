import numpy as np
import sys
import time
import os
import gc

from data_io import train_test_file_reading_with_attrnum
from data_io import list_files
from data_io import init_folder
from data_processing import y_vector_to_matrix
from data_processing import return_data_stru
from data_processing import feature_data_generation
from data_processing import train_test_transpose
from log_io import setup_logger
from object_io import save_obj
from object_io import load_obj
from parameter_proc import read_feature_classification

from tensorflow_cnn import cnn_set_flow_graph
from tensorflow_cnn import cnn_train

from model_setting import return_cnn_setting_from_file
from model_setting import return_cnn_keyword
from classification_results import predict_matrix_with_prob_to_predict_accuracy
from classification_results import f1_value_precision_recall_accuracy
from log_io import init_logging
from ijcnn_2017_cnn import run_feature_projected_ijcnn_fcn



# For projected feature evaluation using projected cnn
# train_x_matrix: train_row, attr_len, attr_num, num_map
def run_feature_projected_cnn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, data_stru, cnn_setting, feature_dict, top_k, saver_file_profix='', class_id=-1, logger=None):
    if logger is None:
        logger = init_logging('')
    method = 'cnn'

    real_num_classes = data_stru.num_classes
    data_stru.num_classes = 2
    cnn_setting.num_classes = 2
    cnn_setting.feature_method = 'none'
    num_classes = 2

    train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    test_row, attr_len, attr_num, input_map = test_x_matrix.shape

    all_predict_matrix = np.zeros(test_row * real_num_classes).reshape(test_row, real_num_classes)

    saver_file = ''
    if class_id == -1:
        min_class = min(train_y_vector)
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
        class_feature = feature_dict[i]
        class_feature = class_feature[0:top_k]
        print ("class: " + str(i))
        print ("number of features: " + str(top_k))
        print ("Top features list: " + str(class_feature))
        logger.info("Top feature list: " + str(class_feature))
        
        temp_train_x_matrix = train_x_matrix[:, :, class_feature, :]
        temp_test_x_matrix = test_x_matrix[:, :, class_feature, :]
        temp_train_y_matrix = y_vector_to_matrix(temp_train_y_vector, num_classes)
        temp_test_y_matrix = y_vector_to_matrix(temp_test_y_vector, num_classes)

        if i == min_class:
            train_x_placeholder, output_y_placeholder, predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_set_flow_graph(data_stru, cnn_setting, input_map, False, logger)
            keep_saver_file = saver_file
        
        saver_file = cnn_setting.temp_obj_folder + class_saver_profix + keep_saver_file + "_top" + str(top_k)
        print saver_file
        class_eval_value, class_train_time, class_test_time, class_predict_prob, fold_saver_file, fold_obj_file = cnn_train(
        temp_train_x_matrix, temp_train_y_matrix, temp_test_x_matrix, temp_test_y_matrix, num_classes, cnn_setting, train_x_placeholder, output_y_placeholder, predict_y_prob, keep_prob_placeholder, keeped_feature_list, saver_file, logger)
        
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
    data_stru.num_classes = real_num_classes
    return all_accuracy, all_f1_value, all_predict_y, all_train_time, all_test_time, all_predict_matrix



def pv_classification_cnn(parameter_file, file_keyword, function_keyword="pv_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, cnn_obj_folder, cnn_temp_folder, cnn_setting_file = read_feature_classification(parameter_file, function_keyword)

    log_folder = init_folder(log_folder)
    cnn_obj_folder = init_folder(cnn_obj_folder)
    cnn_temp_folder = init_folder(cnn_temp_folder)
    
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    file_list = list_files(data_folder)
    obj_list = list_files(obj_folder)
    file_count = 0

    class_column = 0
    header = True

    cnn_setting = return_cnn_setting_from_file(cnn_setting_file)
    cnn_setting.save_obj_folder = cnn_obj_folder
    cnn_setting.temp_obj_folder = cnn_temp_folder
    cnn_setting.eval_method = 'f1'
    init_folder(cnn_obj_folder)
    init_folder(cnn_temp_folder)

    save_obj_folder = obj_folder[:-1] + "_" + method +"_out"
    save_obj_folder = init_folder(save_obj_folder)

    delimiter = ' '
    loop_count = -1
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(class_id) + '_top' + str(top_k) + '_' + method + '.log'
        print "log file: " + log_file
        logger = setup_logger(log_file, 'logger_' + str(loop_count))
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('cnn setting:\n ' + cnn_setting.to_string())
        logger.info('method: ' + method)
        logger.info('============')
        found_obj_file = ''
        for obj_file in obj_list:
            if file_key in obj_file:
                found_obj_file = obj_file
                break
        if found_obj_file == '':
            raise Exception('No obj file found')
        
        #print found_obj_file
        #print cnn_setting.save_obj_folder + file_key + "_" + method +"_projected_result.ckpt"
        #
        found_obj_file = obj_folder + found_obj_file

        feature_dict = load_obj(found_obj_file)[0]
        feature_dict = np.array(feature_dict)
        logger.info("feature array shape: " + str(feature_dict.shape))
        test_file = train_file.replace('train', 'test')

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        
        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)
        data_stru.attr_num = top_k

        if method == 'fcn':
            fold_accuracy, fold_f1_value, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix = run_feature_projected_ijcnn_fcn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, data_stru, cnn_setting, feature_dict, top_k, file_key + '_' + method + '_count' + str(file_count), class_id, logger)
        else:
            fold_accuracy, fold_f1_value, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix = run_feature_projected_cnn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, data_stru, cnn_setting, feature_dict, top_k, file_key + '_' + method + '_count' + str(file_count), class_id, logger)


        logger.info("Fold F1: " + str(fold_f1_value))
        logger.info(method + ' fold training time (sec):' + str(fold_train_time))
        logger.info(method + ' fold testing time (sec):' + str(fold_test_time))
        logger.info(method + ' fold accuracy: ' + str(fold_accuracy))
        logger.info("save obj to " + save_obj_folder + file_key + "_" + method +"_project_" + method +"_result.ckpt")
        save_obj([fold_accuracy, fold_f1_value, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix], save_obj_folder + file_key + "_" + method +"_project_" + method +"_result.ckpt")
