import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import time
from sklearn.metrics import f1_score
from backward_multitime import model_evaluation_cnn
from backward_multitime import model_evaluation_rf
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]),'../src/python/'))
print sys.path
from data_processing import y_vector_to_matrix
from data_processing import data_structure
from data_processing import return_data_stru
#run_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, group_all=False, saver_file_profix='', logger=None):
from tensorflow_cnn import run_cnn
from data_io import init_folder
from data_io import list_files
from object_io import load_obj
from log_io import setup_logger
from model_setting import return_cnn_setting_from_file
from model_setting import return_cnn_keyword
from parameter_proc import read_feature_classification
from data_io import train_test_file_reading

            

def forward_multitime(train_x, train_y, test_x, test_y, n_selected_features, data_key="test", fold_key="", method="cnn", cnn_setting_file = "../../parameters/cnn_model_parameter.txt", logger=None, already_feature=[]):
    """
    This function implements the forward feature selection algorithm based on decision tree

    Input
    -----
    train_x: {3d numpy array matrix}, shape (n_samples, n_features, time_length)
        input data
    train_y: {1d numpy array vector}, shape (n_samples,)
        input class labels
    test_x: {3d numpy array matrix}, shape (n_samples, n_features, time_length)
        input data
    test_y: {1d numpy array vector}, shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    if logger is None:
        log_file = ""
        logger = setup_logger(log_file)

    train_samples, n_features, time_length = train_x.shape

    eval_method = "f1"
    if method == "cnn":
        min_class = min(train_y)
        max_class = max(train_y)
        num_classes = max_class - min_class + 1
        data_stru = data_structure(num_classes, min_class, n_features, time_length)
        cnn_setting = return_cnn_setting_from_file(cnn_setting_file)
        logger.info('cnn setting:\n ' + cnn_setting.to_string())
        saver_file_profix = "../../object/" + data_key + "/forward_wrapper/cnn_model_folder/"
        saver_file_profix = init_folder(saver_file_profix)
        saver_file_profix = saver_file_profix + fold_key
        eval_method = cnn_setting.eval_method
        
    elif method == "rf":
        model = RandomForestClassifier(n_estimators=20, random_state=0)
        
    # selected feature set, initialized to contain all features
    F = already_feature
    count = len(F)
    while count < n_selected_features:
        max_eval_value = -1
        f_score = []
        for i in range(n_features):
            if i not in F:
                F.append(i)
                train_x_tmp = train_x[:, F, :]
                test_x_tmp = test_x[:, F, :]

                if method == "cnn":
                    eval_value, train_run_time, test_run_time, predict_proba, saver_file, feature_list_obj_file = model_evaluation_cnn(train_x_tmp, train_y, test_x_tmp, test_y, data_stru, cnn_setting, saver_file_profix, logger)
                    f_eval_value = eval_value
                elif method == "rf":
                    eval_value, train_run_time, test_run_time = model_evaluation_rf(train_x_tmp, train_y, test_x_tmp, test_y, model, logger)
                    f_eval_value = eval_value
                logger.info("Features With: " + str(F))
                logger.info("Adding Feature " + str(i) + ": ")
                logger.info(method + " " + eval_method + " Value For Feature " + str(i) + ": " + str(f_eval_value))
                logger.info(method +" Training time (sec): " + str(train_run_time))
                logger.info(method + " Testing time (sec): " + str(test_run_time))
                f_score.append(f_eval_value)
                F.pop()
                # record the feature which results in the largest accuracy
                if eval_value > max_eval_value:
                    max_eval_value = eval_value
                    idx = i
        
        logger.info("For iter " + str(count))
        logger.info("Eval score vector: " + str(f_score))
        logger.info("The added attribute is: " + str(idx))
        logger.info("larggest eval value is: " + str(max_eval_value))
        # delete the feature which results in the largest accuracy
        F.append(idx)
        count += 1
    return np.array(F)


def forward_multitime_main(parameter_file="../../parameters/", file_keyword="train_"):
    function_keyword = "forward_wrapper"
    #data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file = read_feature_classification(parameter_file, function_keyword)
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file = read_feature_classification(parameter_file, function_keyword)
    print data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file
    
    if data_keyword == "dsa" or data_keyword == "toy":
        n_selected_features = 15
        num_classes = 19
    elif data_keyword == "rar":
        n_selected_features = 30
        num_classes = 33
    elif data_keyword == "arc" or data_keyword == "fixed_arc":
        n_selected_features = 30
        num_classes = 18
    elif data_keyword == "asl":
        n_selected_features = 6
        num_classes = 95
    else:
        raise Exception("Please fullfill the data basic information first!")


    log_folder = init_folder(log_folder)
    #out_obj_folder = init_folder(out_obj_folder)
    #out_model_folder = init_folder(out_model_folder)
    
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    file_list = list_files(data_folder)

    file_count = 0

    class_column = 0
    header = True

    delimiter = ' '
    loop_count = -1

    ##########
    ###already remove later
    #already_obj_folder = "../../object/" + data_keyword + "/forward_wrapper/"
    #already_obj_list = list_files(already_obj_folder)
    ###end of already remove later
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        #already_obj_file = ""
        already = False
        #for already_obj_file in already_obj_list:
        #    if file_key in already_obj_file and method in already_obj_file:
        #        already = True
        #        break

        ##########
        ###already part
        #if already is True:
        #    already_class_feature = load_obj(already_obj_folder + already_obj_file)[0]
        #else:
        #    log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(class_id) + '_' + method + '.log'
        #    already_class_feature = None
        ###end of already part

        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(class_id) + '_' + method + "_top" + str(n_selected_features) +  '_already' + str(already) +'.log'
        print "log file: " + log_file
    
        logger = setup_logger(log_file, 'logger_' + str(loop_count))
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('method: ' + method)
        logger.info('============')
        

        test_file = train_file.replace('train', 'test')
        

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector = train_test_file_reading(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        n_samples, n_col = train_x_matrix.shape
        train_x_matrix = train_x_matrix.reshape(n_samples, attr_num, attr_len)
        n_samples, n_col = test_x_matrix.shape
        test_x_matrix = test_x_matrix.reshape(n_samples, attr_num, attr_len)
        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))
        
        min_class = min(train_y_vector)
        max_class = max(train_y_vector) + 1
        for c in range(min_class, max_class):
            logger.info("Class: " + str(c))
            already_feature = []
            #if already_class_feature is not None:
            #    class_already = already_class_feature[c, :]
            #    for already_f in class_already:
            #        already_feature.append(already_f)
            #    logger.info("already features: " +file_key + " with class " + str(c) + ": " + str(already_feature))
            temp_train_y_vector = np.where(train_y_vector == c, 1, 0)
            temp_test_y_vector = np.where(test_y_vector == c, 1, 0)
            #print already_feature
            top_features = forward_multitime(train_x_matrix, temp_train_y_vector, test_x_matrix, temp_test_y_vector, n_selected_features, data_keyword, file_key, method, cnn_setting_file, logger, already_feature)
            logger.info("Top Features For Class " +str(c) + ": " + str(top_features))
            logger.info("End Of Class: " + str(c))


if __name__ == "__main__":
    parameter_file = "../../parameters/pv_classification.txt"
    argv_array = sys.argv
    file_keyword="train_"
    if len(argv_array)==2:
        file_keyword = file_keyword + str(argv_array[1])
    forward_multitime_main(parameter_file, file_keyword)
