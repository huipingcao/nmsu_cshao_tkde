import sys
import time
import numpy as np
from log_io import setup_logger
from data_io import list_files
from data_io import file_reading
from data_io import x_y_spliting
from data_io import init_folder
from object_io import save_obj
from object_io import load_obj

from feature_evaluation import rf_feature_extraction
from feature_evaluation import lda_feature_extraction
from data_processing import return_data_stru
from tkde_2005_pca import computeDCPC
from tkde_2005_pca import clever_rank
import operator


def project_cnn_feature_combined_cpca_analysis(feature_matrix, y_vector, logger=None):
    if logger is None:
        logger = setup_logger('')
    threshold = 0.9

    feature_matrix = np.squeeze(feature_matrix)
    num_instance, num_attribute, num_map = feature_matrix.shape
    predict = True
    rf_time = 0
    lda_time = 0
    map_attr_imp_matrix = []
    success_count = 0

    label_index = np.where(y_vector==1)[0]
    label_x_matrix = feature_matrix[label_index, :, :]
    logger.info("x matrix tran before shape: " + str(label_x_matrix.shape))
    start_time = time.time()
    cpca_matrix = computeDCPC(label_x_matrix, threshold)
    attr_score = clever_rank(cpca_matrix, logger)
    sorted_dict = sorted(attr_score.items(), key=operator.itemgetter(1), reverse=True)
    sorted_attr = []
    for item in sorted_dict:
        sorted_attr.append(item[0])
    run_time = time.time() - start_time
    print cpca_matrix.shape
    print sorted_attr
    return sorted_attr, run_time



def project_cnn_feature_combined_rf_analysis(feature_matrix, y_vector, logger=None, rf_estimator=50):
    if logger is None:
        logger = setup_logger('')
    feature_matrix = np.squeeze(feature_matrix)
    num_instance, num_attribute, num_map = feature_matrix.shape
    predict = True
    skip_count = 0
    rf_time = 0
    lda_time = 0
    map_attr_imp_matrix = []
    success_count = 0

    feature_matrix_2d = feature_matrix.reshape(num_instance, num_attribute*num_map)
    start_time = time.time()
    rf_feature_value_vector, rf_model, rf_f1_value, rf_run_time = rf_feature_extraction(feature_matrix_2d, y_vector, predict, logger, rf_estimator)
    rf_time = rf_time + time.time() - start_time
    
    rf_feature_value_vector = rf_feature_value_vector.reshape(num_attribute, num_map)
    rf_feature_value_vector = np.sum(rf_feature_value_vector, axis=1)
    sum_value = sum(rf_feature_value_vector)
    rf_feature_value_vector = rf_feature_value_vector/float(sum_value)
    rf_feature_value_vector = rf_feature_value_vector * rf_f1_value
    rf_class_attr_list = map_attr_imp_analysis(rf_feature_value_vector, logger)

    logger.info("rf feature value: " + str(rf_feature_value_vector.shape))
    logger.info("rf f1 value: " + str(rf_f1_value))
    logger.info("rf only attr:" + str(rf_class_attr_list))

    feature_value_vector = rf_feature_value_vector
    return feature_value_vector, rf_time + lda_time



# This function put all the num_map together and add the absolute values on all the 20 values
def project_cnn_feature_combined_lda_analysis(feature_matrix, y_vector, logger=None, rf_estimator=50):
    if logger is None:
        logger = setup_logger('')
    feature_matrix = np.squeeze(feature_matrix)
    num_instance, num_attribute, num_map = feature_matrix.shape
    predict = True
    skip_count = 0
    rf_time = 0
    lda_time = 0
    map_attr_imp_matrix = []
    success_count = 0

    feature_matrix_2d = feature_matrix.reshape(num_instance, num_attribute*num_map)
    start_time = 0
    lda_feature_value_vector, lda_model, lda_f1_value, lda_run_time = lda_feature_extraction(feature_matrix_2d, y_vector, predict, logger)
    lda_time = time.time() - start_time
    
    lda_feature_value_vector = lda_feature_value_vector.reshape(num_attribute, num_map)
    lda_feature_value_vector = np.sum(lda_feature_value_vector, axis=1)
    sum_value = sum(lda_feature_value_vector)
    lda_feature_value_vector = lda_feature_value_vector/float(sum_value)
    lda_class_attr_list = map_attr_imp_analysis(lda_feature_value_vector, logger)

    logger.info("lda feature value: " + str(lda_feature_value_vector.shape))
    logger.info("lda f1 value: " + str(lda_f1_value))
    logger.info("lda only attr:" + str(lda_class_attr_list))

    feature_value_vector = lda_feature_value_vector
    return feature_value_vector, rf_time + lda_time



# This function put all the num_map together and add the absolute values on all the 20 values
def project_cnn_feature_combined_rf_lda_analysis(feature_matrix, y_vector, logger=None, rf_estimator=50):
    if logger is None:
        logger = setup_logger('')
    #feature_matrix = np.squeeze(feature_matrix)
    num_instance, num_attribute, num_map = feature_matrix.shape
    predict = True
    skip_count = 0
    rf_time = 0
    lda_time = 0
    map_attr_imp_matrix = []
    success_count = 0

    feature_matrix_2d = feature_matrix.reshape(num_instance, num_attribute*num_map)
    start_time = 0
    rf_feature_value_vector, rf_model, rf_f1_value, rf_run_time = rf_feature_extraction(feature_matrix_2d, y_vector, predict, logger, rf_estimator)
    rf_time = rf_time + time.time() - start_time
    start_time = 0
    lda_feature_value_vector, lda_model, lda_f1_value, lda_run_time = lda_feature_extraction(feature_matrix_2d, y_vector, predict, logger)
    lda_time = time.time() - start_time
    #print rf_feature_value_vector
    #print np.sum(rf_feature_value_vector)
    #print lda_feature_value_vector
    #print np.sum(lda_feature_value_vector)
    
    rf_feature_value_vector = rf_feature_value_vector.reshape(num_attribute, num_map)
    rf_feature_value_vector = np.sum(rf_feature_value_vector, axis=1)
    sum_value = sum(rf_feature_value_vector)
    rf_feature_value_vector = rf_feature_value_vector/float(sum_value)
    rf_feature_value_vector = rf_feature_value_vector * rf_f1_value
    rf_class_attr_list = map_attr_imp_analysis(rf_feature_value_vector, logger)

    logger.info("rf feature value: " + str(rf_feature_value_vector))
    logger.info("rf f1 value: " + str(rf_f1_value))
    logger.info("rf only attr:" + str(rf_class_attr_list))
    
    feature_value_vector = rf_feature_value_vector
    if lda_feature_value_vector is not None:
        lda_feature_value_vector = lda_feature_value_vector.reshape(num_attribute, num_map)
        lda_feature_value_vector = np.sum(lda_feature_value_vector, axis=1)
        sum_value = sum(lda_feature_value_vector)
        lda_feature_value_vector = lda_feature_value_vector/float(sum_value)
        lda_feature_value_vector = lda_feature_value_vector * lda_f1_value
        lda_class_attr_list = map_attr_imp_analysis(lda_feature_value_vector, logger)
        lda_max = max(lda_feature_value_vector)
        logger.info("lda only attr:" + str(lda_class_attr_list))
        logger.info("lda feature value: " + str(lda_feature_value_vector))
        logger.info("lda f1 value: " + str(lda_f1_value))
        logger.info("max lda value: " + str(lda_max))
        if lda_max < 0.9:
            feature_value_vector = feature_value_vector + lda_feature_value_vector

    #if rf_f1_value > lda_f1_value:
    #    feature_value_vector = rf_feature_value_vector
    #elif rf_f1_value < lda_f1_value:
    #    feature_value_vector = lda_feature_value_vector
    #else:
    #    feature_value_vector = rf_feature_value_vector + lda_feature_value_vector
    
    sum_value = sum(feature_value_vector)
    feature_value_vector = feature_value_vector/float(sum_value)
    class_attr_list = map_attr_imp_analysis(feature_value_vector, logger)
    logger.info("overall rf and lda feature value: " + str(feature_value_vector))
    logger.info("rf and lda attr:" + str(class_attr_list))
    return feature_value_vector, rf_time + lda_time

# This function put all the num_map together and do majority vote
def project_cnn_feature_combined_rf_lda_analysis_majority_vote(feature_matrix, y_vector, logger=None, rf_estimator=50):
    if logger is None:
        logger = setup_logger('')
    feature_matrix = np.squeeze(feature_matrix)
    num_instance, num_attribute, num_map = feature_matrix.shape
    predict = True
    skip_count = 0
    rf_time = 0
    lda_time = 0
    map_attr_imp_matrix = []
    success_count = 0
    for i in range(0, num_map):
        map_feature_matrix = feature_matrix[:, :, i]
        if np.any(map_feature_matrix) is False:
            skip_count = skip_count + 1
            continue
        rf_lda_avg_acc = 0
        #feature_value_vector = []
        start_time = 0
        rf_feature_value_vector, rf_model, rf_f1_value, rf_run_time = rf_feature_extraction(map_feature_matrix, y_vector, predict, logger, rf_estimator)
        rf_time = rf_time + time.time() - start_time
        if rf_model is not None:
            rf_lda_avg_acc = rf_feature_value_vector
            feature_value_vector = rf_feature_value_vector * rf_f1_value
        start_time = 0
        lda_feature_vector, lda_model, lad_averaged_acc, lad_run_time = lda_feature_extraction(map_feature_matrix, y_vector, predict, logger)
        lda_time = lda_time + time.time() - start_time

        if lda_model is not None:
            rf_lda_avg_acc = rf_lda_avg_acc + lda_feature_vector
            feature_value_vector = feature_value_vector + lda_feature_vector * lad_averaged_acc
            success_count = success_count + 1
        map_attr_imp_matrix.append(feature_value_vector)
    map_attr_imp_matrix = np.array(map_attr_imp_matrix)
    logger.info("success count: " + str(success_count))
    return map_attr_imp_matrix, rf_time + lda_time


# This function only keep the highest rf and lda accuracy on all num_map (20)
def project_cnn_feature_combined_rf_lda_analysis_with_highest(feature_matrix, y_vector, logger=None, rf_estimator=50):
    if logger is None:
        logger = setup_logger('')
    feature_matrix = np.squeeze(feature_matrix)
    num_instance, num_attribute, num_map = feature_matrix.shape
    predict = True
    skip_count = 0
    rf_time = 0
    lda_time = 0
    keep_avg_acc = -1
    map_attr_imp_matrix = []
    success_count = 0
    for i in range(0, num_map):
        map_feature_matrix = feature_matrix[:, :, i]
        if np.any(map_feature_matrix) is False:
            skip_count = skip_count + 1
            continue
        rf_lda_avg_acc = 0
        #feature_value_vector = []

        start_time = 0
        rf_feature_value_vector, rf_model, rf_f1_value, rf_run_time = rf_feature_extraction(map_feature_matrix, y_vector, predict, logger, rf_estimator)
        rf_time = rf_time + time.time() - start_time
        if rf_model is not None:
            rf_lda_avg_acc = rf_feature_value_vector
            feature_value_vector = rf_feature_value_vector * rf_f1_value
        start_time = 0
        lda_feature_vector, lda_model, lad_averaged_acc, lad_run_time = lda_feature_extraction(map_feature_matrix, y_vector, predict, logger)
        lda_time = lda_time + time.time() - start_time

        if lda_model is not None:
            rf_lda_avg_acc = rf_lda_avg_acc + lda_feature_vector
        success_count = success_count + 1
        #rf_lda_avg_acc = rf_f1_value + lad_averaged_acc
        if keep_avg_acc < rf_lda_avg_acc:
            keep_avg_acc = rf_lda_avg_acc
            feature_value_vector = feature_value_vector + lda_feature_vector * lad_averaged_acc
            map_attr_imp_matrix = [feature_value_vector]
        elif keep_avg_acc == rf_lda_avg_acc:
            map_attr_imp_matrix.append(feature_value_vector)
    map_attr_imp_matrix = np.array(map_attr_imp_matrix)
    logger.info("success count: " + str(success_count))
    logger.info("highest acc:" + str(keep_avg_acc))
    return map_attr_imp_matrix, rf_time + lda_time


def map_attr_imp_analysis(map_attr_imp_matrix, logger=None):
    if logger is None:
        logger = setup_logger('')
    logger.info(map_attr_imp_matrix.shape)
    if map_attr_imp_matrix.ndim == 1:
        attr_imp = map_attr_imp_matrix
        sort_index = np.argsort(attr_imp)
        imp_value = 0
        norm_imp = np.zeros(len(attr_imp))
        for index in sort_index:
            norm_imp[index] = imp_value
            imp_value = imp_value + 1
        return np.argsort(norm_imp)[::-1]
    num_map, num_attr = map_attr_imp_matrix.shape

    norm_all_imp = []
    for i in range(0, num_map):
        attr_imp = map_attr_imp_matrix[i, :]
        sort_index = np.argsort(attr_imp)
        imp_value = 0
        norm_imp = np.zeros(len(attr_imp))
        for index in sort_index:
            norm_imp[index] = imp_value
            imp_value = imp_value + 1
        if len(norm_all_imp) == 0:
            norm_all_imp = norm_imp
        else:
            norm_all_imp = norm_all_imp + norm_imp
    return np.argsort(norm_all_imp)[::-1]


def obj_processing(feature_obj):
    if len(feature_obj) == 1:
        return np.squeeze(np.array(feature_obj))
    else:
        ret_obj = []
        for obj in feature_obj:
            obj = np.squeeze(obj)
            if len(ret_obj) == 0:
                ret_obj = obj
            else:
                ret_obj = np.concatenate((ret_obj, obj), axis=0)
        return np.array(ret_obj)


def run_cnn_projected_feature_analysis(feature_folder, class_id, data_folder, data_file_keyword, method="rf_lda", log_folder='./'):
    data_file_list = list_files(data_folder)
    feature_file_list = list_files(feature_folder)
    out_obj_folder = feature_folder[:-1] + "_" + method
    out_obj_folder = init_folder(out_obj_folder)
    class_column = 0

    for train_file in data_file_list:
        if data_file_keyword not in train_file:
            continue
        data_key = train_file.replace('.txt', '')
        data_matrix, attr_num = file_reading(data_folder + train_file)
        train_x_matrix, train_y_vector = x_y_spliting(data_matrix, class_column)
        if class_id < 0:
            min_class = min(train_y_vector)
            max_class = max(train_y_vector) + 1
        else:
            min_class = class_id
            max_class = min_class + 1
        log_file = data_key + "_" + method + "_min" + str(min_class) + "_max" + str(max_class) + ".log"
        logger = setup_logger(log_folder + log_file)
        logger.info('data file: '+ train_file)
        out_obj_file = data_key + "_" + method + "_min" + str(min_class) + "_max" + str(max_class) + ".obj"
        out_obj_matrix = []
        for label in range(min_class, max_class):
            logger.info("class: " + str(label))
            feature_key = "_class" + str(label) + "_"
            for feature_file in feature_file_list:
                if data_key not in feature_file or feature_key not in feature_file:
                    continue
                logger.info("feature file: " + feature_file)
                feature_obj = load_obj(feature_folder + feature_file)
                train_feature = obj_processing(feature_obj[0])
                logger.info("train feature shape: " + str(train_feature.shape))
                class_train_y = np.where(train_y_vector == label, 1, 0)
                logger.info("feature method: " + str(method))
                if method == "rf_lda":
                    class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_rf_lda_analysis(train_feature, class_train_y, logger)
                elif method == "rf":
                    class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_rf_analysis(train_feature, class_train_y, logger)
                elif method == "lda":
                    class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_lda_analysis(train_feature, class_train_y, logger)
                elif method == "cpca":
                    class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_cpca_analysis(train_feature, class_train_y, logger)
                if method == "cpca":
                    class_attr_list = class_attr_imp_matrix
                else:
                    logger.info("class attr imp matrix shape: " + str(class_attr_imp_matrix.shape))
                    class_attr_list = map_attr_imp_analysis(class_attr_imp_matrix, logger)
                logger.info(class_attr_list)
                out_obj_matrix.append(class_attr_list)
        out_obj_matrix = np.array(out_obj_matrix)
        logger.info("out obj to: " + out_obj_folder + out_obj_file)
        logger.info(out_obj_matrix.shape)
        save_obj([out_obj_matrix], out_obj_folder + out_obj_file)



def pv_cnn_evaluation_main(data_keyword="toy", method="rf_lda", file_keyword="train_", class_id=-1):
    feature_folder = "../../object/" + data_keyword + "/pv_cnn_generation/cnn_obj_folder/"
    data_folder = "../../data/" + data_keyword + "/"
    log_folder = "../../log/"+data_keyword+"/pv_cnn_evaluation/"
    log_folder = init_folder(log_folder)
    run_cnn_projected_feature_analysis(feature_folder, class_id, data_folder, file_keyword, method, log_folder)

if __name__ == '__main__':
    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train_'
    projected = True
    len_argv_array = len(argv_array)

    try:
        data_key = argv_array[1]
        method = argv_array[2]
        class_id = -1
    except Exception:
        print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print ("REQURIED PARAMETERS MISSING")
        print ("python pv_cnn_evaluation #data_keyword #method #file_keyword #class_id")
        print ("#data_keyword and #method are required")
        print ("#file_keyword and #class_id are optional")
        print ("Without #file_keyword, it will loop all files under the data folder")
        print ("Without #class_id, it will loop all classes")
        exit()


    if len_argv_array > 3:
        try:
            val = int(argv_array[3])
            file_keyword = file_keyword + str(val)
            if len_argv_array > 4:
                try:
                    class_id = int(argv_array[4])
                except Exception:
                    print ("Selected class id should be int")
        except Exception:
            print ("That's not an int!")
    pv_cnn_evaluation_main(data_key, method, file_keyword, class_id)
