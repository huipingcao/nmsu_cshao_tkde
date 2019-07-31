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
from log_io import setup_logger
from object_io import save_obj
from object_io import load_obj
from parameter_proc import read_feature_classification

#from tensorflow_cnn import run_projected_feature_cnn

from model_setting import return_cnn_setting_from_file
from model_setting import return_cnn_keyword
from classification_results import multiple_f1_value_precision_recall_accuracy

from projected_feature_cnn_main import run_feature_projected_cnn
from data_processing import train_test_transpose



def global_classification_main(parameter_file, file_keyword):
    function_keyword="global_classification"
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, cnn_obj_folder, cnn_temp_folder, cnn_setting_file = read_feature_classification(parameter_file, function_keyword)
    
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

    all_result_matrix = np.zeros((10, num_classes))

    train_file_vector = []
    prediction_matrix = []
    f1_value_matrix = []
    accuracy_vector = []
    delimiter = ' '
    all_accuracy = 0
    all_train_time = 0
    all_test_time = 0
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
        continue
        found_obj_file = ''
        for obj_file in obj_list:
            if file_key in obj_file:
                found_obj_file = obj_file
                break
        if found_obj_file == '':
            raise Exception('No obj file found')
        
        print found_obj_file
        print cnn_setting.save_obj_folder + file_key + "_" + method +"_projected_result.ckpt"
        #
        found_obj_file = obj_folder + found_obj_file

        feature_dict = load_obj(found_obj_file)[0]
        feature_dict = np.array(feature_dict)
        logger.info("feature array shape: " + str(feature_dict.shape))
        
        test_file = train_file.replace('train', 'test')

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        
        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)
        data_stru.attr_num = top_k
        fold_accuracy, fold_avg_eval, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix = run_feature_projected_cnn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, data_stru, cnn_setting, feature_dict, top_k, file_key + '_count' + str(file_count), class_id, logger)

        prediction_matrix.append(fold_predict_y)
        logger.info("Fold F1: " + str(fold_f1_value_list))
        accuracy_vector.append(fold_accuracy)
        all_accuracy = all_accuracy + fold_accuracy
        all_train_time = all_train_time + fold_train_time
        all_test_time = all_test_time + fold_test_time
        logger.info(method + ' fold accuracy: ' + str(fold_accuracy))
        logger.info(method + ' fold training time (sec):' + str(fold_train_time))
        logger.info(method + ' fold testing time (sec):' + str(fold_test_time))
        save_obj([fold_accuracy, fold_avg_eval, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix], save_obj_folder + file_key + "_" + method +"_global_cnn_result.ckpt")
    


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

    parameter_file = '../../parameters/global_feature_classification.txt'
    global_classification_main(parameter_file, file_keyword)
    #
