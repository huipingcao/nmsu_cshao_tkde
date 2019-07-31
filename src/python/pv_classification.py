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

from classification_results import predict_matrix_with_prob_to_predict_accuracy
from classification_results import f1_value_precision_recall_accuracy

from sklearn_classification import run_knn
from sklearn_classification import run_rf
from sklearn_classification import run_libsvm
from data_processing import train_test_transpose
from projected_feature_cnn_main import projected_cnn_classification_main
from log_io import init_logging



# For projected feature evaluation using projected cnn
# train_x_matrix: train_row, attr_len, attr_num, input_map
def run_feature_projected_classification(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, top_k, method, class_id=-1, logger=None):
    if logger == None:
        logger = init_logging('')

    train_row, attr_len, attr_num, input_map = train_x_matrix.shape
    test_row, attr_len, attr_num, input_map = test_x_matrix.shape
    real_num_classes, attr_num = feature_array.shape
    all_predict_matrix = np.zeros(test_row * real_num_classes).reshape(test_row, real_num_classes)

    feature_col = attr_len * top_k * input_map

    if class_id == -1:
        min_class = min(train_y_vector)
        max_class = max(train_y_vector) + 1
    else:
        min_class = class_id
        max_class = class_id + 1

    n_neighbors = 1
    samples_leaf = 20
    prob = True

    all_f1_value = []
    all_train_time = []
    all_test_time = []
    #min_class = 9
    for i in range(min_class, max_class):
        logger.info('class: ' + str(i))
        temp_train_y_vector = np.where(train_y_vector == i, 1, 0)
        temp_test_y_vector = np.where(test_y_vector == i, 1, 0)

        fold_positive_len = len(np.where(temp_train_y_vector == 1)[0])
        fold_negative_len = len(temp_train_y_vector) - fold_positive_len

        logger.info("=====")
        logger.info("positive class labels length: " + str(fold_positive_len))
        logger.info("negative class labels length: " + str(fold_negative_len))
        class_feature = feature_array[i]
        class_feature = class_feature[0:top_k]
        logger.info("feature list: " + str(class_feature))
        
        temp_train_x_matrix = train_x_matrix[:, :, class_feature, :]
        temp_test_x_matrix = test_x_matrix[:, :, class_feature, :]
        temp_train_x_matrix = temp_train_x_matrix.reshape(train_row, feature_col)
        temp_test_x_matrix = temp_test_x_matrix.reshape(test_row, feature_col)

        if method == 'knn':
            class_accuracy, class_predict_y, class_predict_prob, class_train_time, class_test_time = run_knn(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, n_neighbors, prob)
        elif method == 'rf':
            class_accuracy, class_predict_y, class_predict_prob, class_train_time, class_test_time = run_rf(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, samples_leaf, prob)
        elif method == 'libsvm':
            class_accuracy, class_predict_y, class_predict_prob, class_train_time, class_test_time = run_libsvm(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, logger, prob, '', True)
        

        class_accuracy, precision, recall, class_f1, tp, fp, tn, fn = f1_value_precision_recall_accuracy(class_predict_y, temp_test_y_vector, 1)

        logger.info(method + " f1 for class "+ str(i) + ": " + str(class_f1))
        logger.info(method + " accuracy for class "+ str(i) + ": " + str(class_accuracy))

        all_f1_value.append(class_f1)
        all_train_time.append(class_train_time)
        all_test_time.append(class_test_time)
        all_predict_matrix[:, i] = class_predict_prob[:, 1]
        #if i > 2:
        #    break
    all_accuracy, all_predict_y = predict_matrix_with_prob_to_predict_accuracy(all_predict_matrix, test_y_vector)
    return all_accuracy, all_f1_value, all_predict_y, all_train_time, all_test_time, all_predict_matrix


def projected_classification_main(parameter_file, file_keyword, function_keyword="projected_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, cnn_obj_folder, cnn_temp_folder, cnn_setting_file = read_feature_classification(parameter_file, function_keyword)
    log_folder = init_folder(log_folder)
    if method == 'cnn':
        return projected_cnn_classification_main(parameter_file, file_keyword)
    

    print data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, cnn_obj_folder, cnn_temp_folder, cnn_setting_file
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)
    print obj_folder
    file_list = list_files(data_folder)
    obj_list = list_files(obj_folder)

    class_column = 0
    header = True

    save_obj_folder = obj_folder[:-1] + "_" + method + "_out"
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
        logger.info('method: ' + method)
        logger.info('============')

        found_obj_file = ''
        for obj_file in obj_list:
            if file_key in obj_file:
                found_obj_file = obj_file
                break
        if found_obj_file == '':
            raise Exception('No obj file found')
        
        print found_obj_file
        found_obj_file = obj_folder + found_obj_file

        feature_array = load_obj(found_obj_file)[0]
        feature_array = np.array(feature_array)
        logger.info("feature array shape: " + str(feature_array.shape))
        
        test_file = train_file.replace('train', 'test')

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        
        if loop_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)

        data_stru.attr_num = top_k
        fold_accuracy, fold_f1_value, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix = run_feature_projected_classification(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, top_k, method, class_id, logger)

        logger.info("Fold F1: " + str(fold_f1_value))
        logger.info(method + ' fold training time (sec):' + str(fold_train_time))
        logger.info(method + ' fold testing time (sec):' + str(fold_test_time))
        logger.info(method + ' fold accuracy: ' + str(fold_accuracy))
        logger.info("save obj to " + save_obj_folder + file_key + "_" + method +"_project_" + method +"_result.ckpt")
        save_obj([fold_accuracy, fold_f1_value, fold_predict_y, fold_train_time, fold_test_time, fold_predict_matrix], save_obj_folder + file_key + "_" + method +"_project_" + method +"_result.ckpt")
    

    

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

    parameter_file = '../../parameters/projected_feature_classification.txt'
    projected_classification_main(parameter_file, file_keyword)
    #
