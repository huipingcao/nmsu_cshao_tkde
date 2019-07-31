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
from parameter_proc import read_all_feature_classification

from tensorflow_cnn import run_cnn

from model_setting import return_cnn_setting_from_file
from model_setting import return_cnn_keyword
from classification_results import predict_matrix_with_prob_to_predict_accuracy
from classification_results import f1_value_precision_recall_accuracy
from log_io import init_logging

from ijcnn_2017_cnn import run_feature_projected_ijcnn_fcn



# This is a multi-class classification using CNN model. Using Accuracy instead of F1 as measurement
# Just classification, no need to store the output objects
def cnn_classification_main(parameter_file, file_keyword, function_keyword="cnn_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file = read_all_feature_classification(parameter_file, function_keyword)

    print data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file

    log_folder = init_folder(log_folder)
    out_obj_folder = init_folder(out_obj_folder)
    out_model_folder = init_folder(out_model_folder)
    
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    file_list = list_files(data_folder)
    obj_list = list_files(obj_folder)
    file_count = 0

    class_column = 0
    header = True

    cnn_setting = return_cnn_setting_from_file(cnn_setting_file)
    cnn_setting.out_obj_folder = out_obj_folder
    cnn_setting.out_model_folder = out_model_folder
    cnn_setting.feature_method = 'none'
    cnn_key = return_cnn_keyword(cnn_setting)
    init_folder(out_obj_folder)
    init_folder(out_model_folder) 
    group_all = False
    result_obj_folder = obj_folder + method +"_result_folder"
    result_obj_folder = init_folder(result_obj_folder)

    delimiter = ' '
    loop_count = -1
    saver_file_profix = ""
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        saver_file_profix = file_key
        test_file = train_file.replace('train', 'test')

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        min_class = min(train_y_vector)
        max_class = max(train_y_vector)
        num_classes = max_class - min_class + 1
        if cnn_setting.eval_method == "accuracy":
            cnn_eval_key = "acc"
        elif num_classes > 2:
            cnn_eval_key = "acc_batch"
        else:
            cnn_eval_key = "f1"
        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(min_class)+"_" + str(max_class) + "_act" + str(cnn_setting.activation_fun) + "_" + cnn_eval_key + '.log'
    
        print "log file: " + log_file
    
        logger = setup_logger(log_file, 'logger_' + str(loop_count))
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('cnn setting:\n ' + cnn_setting.to_string())
        logger.info('method: ' + method)
        logger.info('============')
        #train_y_vector[50:80] = 1
        #test_y_vector[30:40] = 1

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)
        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))

        logger.info(train_x_matrix[0, 0:3, 0:2, 0])
        logger.info(test_x_matrix[0, 0:3, 0:2, 0])

        train_y_matrix = y_vector_to_matrix(train_y_vector, num_classes)
        test_y_matrix = y_vector_to_matrix(test_y_vector, num_classes)

        feature_dict = None
        top_k = -1
        model_save_file = file_key + '_count' + str(file_count) + '_' + method 

        cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file, relu_base_array = run_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, group_all, saver_file_profix, logger)

        logger.info("Fold eval value: " + str(cnn_eval_value))
        logger.info(method + ' fold training time (sec):' + str(train_run_time))
        logger.info(method + ' fold testing time (sec):' + str(test_run_time))
        logger.info("save obj to " + saver_file)
        

    

if __name__ == '__main__':
    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train_'
    projected = True
    len_argv_array = len(argv_array)
    if len_argv_array > 1:
        try:
            val = int(argv_array[1])
            file_keyword = file_keyword + argv_array[1]
        except ValueError:
           print("That's not an int!")

    parameter_file = '../../parameters/all_feature_classification.txt'
    cnn_classification_main(parameter_file, file_keyword)
    #
