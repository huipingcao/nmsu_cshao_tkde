import numpy as np
import sys
import time
import os
import gc
from sklearn.metrics import accuracy_score

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

from tensorflow_cnn import load_model
from tensorflow_cnn import load_model_predict

from model_setting import return_cnn_setting_from_file
from model_setting import return_cnn_keyword
from classification_results import predict_matrix_with_prob_to_predict_accuracy
from classification_results import multiple_f1_value_precision_recall_accuracy
from log_io import init_logging

from ijcnn_2017_cnn import run_feature_projected_ijcnn_fcn



# For projected feature evaluation using projected cnn
# train_x_matrix: train_row, attr_len, attr_num, num_map
def run_load_predict_cnn(fold_keyword, model_saved_folder, feature_array, top_k, test_x_matrix, test_y_vector, data_stru, cnn_setting, group_all=True, save_obj_folder="./", logger=None):
    if logger == None:
        logger = init_logging('')
    
    real_num_classes = data_stru.num_classes
    model_list = list_files(model_saved_folder)
    data_stru.num_classes = 2
    
    load_time = 0
    test_time = 0
    multi_predict = []
    for c in range(real_num_classes):
        logger.info("Class: " + str(c))
        class_keyword = "class" + str(c) + "_"
        found_model_file = ""
        for model_file in model_list:
            if ".index" not in model_file:
                continue
            if fold_keyword not in model_file:
                continue
            if class_keyword not in model_file:
                continue
            found_model_file = model_file.replace(".index", "")
            print (found_model_file)
            break
    
        if found_model_file == "":
            raise Exception("Model for " + class_keyword + " and " + fold_keyword + " Not Found!!!")
        else:
            found_model_file = model_saved_folder + found_model_file
        class_feature = feature_array[c]
        class_feature = class_feature[0:top_k]
        logger.info("model file: " + str(model_saved_folder + found_model_file))
        logger.info("feature list: " + str(class_feature))
        
        temp_test_x_matrix = test_x_matrix[:, :, class_feature, :]
        logger.info("In run_load_predict_cnn: " + str(temp_test_x_matrix.shape))
        start_time = time.time()
        cnn_session, predict_y_proba, train_x_placeholder, keep_prob_placeholder = load_model(found_model_file, data_stru, cnn_setting, group_all, logger)
        load_time = load_time + time.time() - start_time
        start_time = time.time()
        cnn_predict_proba = load_model_predict(cnn_session, temp_test_x_matrix, predict_y_proba, train_x_placeholder, keep_prob_placeholder)
        #print (cnn_predict_proba[0:10, :])
        test_time = test_time + time.time() - start_time
        multi_predict.append(cnn_predict_proba[:, 1])
        cnn_session.close()
    
    multi_predict = np.array(multi_predict)
    #print multi_predict[0:2, 5:11]
    multi_predict_vector = np.argmax(multi_predict, axis=0)
    save_obj_file = save_obj_folder + fold_keyword + "_" + str(top_k) + ".out"
    save_obj([multi_predict], save_obj_file)
    logger.info("output obj saved to: " + save_obj_file)
    logger.info("multi predict matrix shape: " + str(multi_predict.shape))
    logger.info("multi predict vector shape: " + str(multi_predict_vector.shape))
    #print (str(multi_predict_vector[0:10]))
    logger.info("test y vector: " + str(test_y_vector.shape))
    #print (str(test_y_vector[0:10]))
    acc = accuracy_score(test_y_vector, multi_predict_vector)
    data_stru.num_classes = real_num_classes
    acc1, f1_list = multiple_f1_value_precision_recall_accuracy(multi_predict_vector, test_y_vector, logger)
    if acc != acc1:
        raise Exception("check accuracy")
    return acc, f1_list, load_time, test_time


# Load trained cnn model and do prediction
# Load all projected features from each class and do a overall prediction based on the highest probability
def multi_projected_cnn_classification_main(parameter_file, file_keyword, function_keyword="multi_proj_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, cnn_obj_folder, cnn_temp_folder, cnn_setting_file = read_feature_classification(parameter_file, function_keyword)

    obj_keyword = obj_folder.split('/')[-2]
    
    model_saved_folder = "../../object/" + data_keyword + "/projected_classification/" + obj_keyword + "_top" + str(top_k) + "_cnn_model_folder/"
    print obj_keyword
    print cnn_obj_folder
    print model_saved_folder
    top_keyword = "_top" + str(top_k) + "."
    group_all = False

    log_folder = init_folder(log_folder)
    #cnn_obj_folder = init_folder(cnn_obj_folder)
    #cnn_temp_folder = init_folder(cnn_temp_folder)
    
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
    #init_folder(cnn_obj_folder)
    #init_folder(cnn_temp_folder) 

    save_obj_folder = "../../object/" + data_keyword + "/" + function_keyword + "/" + obj_keyword + "/" 
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
        #
        found_obj_file = obj_folder + found_obj_file

        feature_dict = load_obj(found_obj_file)[0]
        feature_dict = np.array(feature_dict)
        logger.info("feature array shape: " + str(feature_dict.shape))
        
        test_file = train_file.replace('train', 'test')

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)

        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))
            logger.info("topk: " + str(top_k) )
        data_stru.attr_num = top_k
        fold_accuracy, fold_f1_list, fold_load_time, fold_test_time = run_load_predict_cnn(file_key, model_saved_folder, feature_dict, top_k, test_x_matrix, test_y_vector, data_stru, cnn_setting, group_all, save_obj_folder, logger)

        logger.info("Fold ACC: " + str(fold_accuracy))
        logger.info("Fold F1 list: " + str(fold_f1_list))
        logger.info(method + ' fold training time (sec):' + str(fold_load_time))
        logger.info(method + ' fold testing time (sec):' + str(fold_test_time))
    
    
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
    function_keyword = "multi_proj_feature_classification"
    multi_projected_cnn_classification_main(parameter_file, file_keyword, function_keyword)

