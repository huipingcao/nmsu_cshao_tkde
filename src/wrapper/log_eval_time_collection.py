import logging
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]),'../src/fileio/'))
from data_io import list_files
from data_io import init_folder
from data_io import write_to_excel
from object_io import save_obj



def results_from_file(file_name, train_time_key, test_time_key):
    feature_dict = {}
    min_class = 100
    max_class = -1
    train_time = 0
    test_time = 0

    with open(file_name) as f:
        value_vector = []
        for line in f:
            if train_time_key in line:
                line_array = line.split(':')
                train_time = train_time + float(line_array[-1].strip())
            elif test_time_key in line:
                line_array = line.split(':')
                test_time = test_time + float(line_array[-1].strip())
    
    return train_time, test_time


def results_from_folder(folder_name, file_keyword, train_time_keyword, test_time_keyword, fold_count=10):
    file_list = list_files(folder_name)
    file_count = 0
    train_list = []
    test_list = []
    for fold_id in range(fold_count):
        fold_key = "train_" + str(fold_id) + "_"
        for file_name in file_list:
            if file_name.startswith('.'):
                continue
            if fold_key not in file_name:
                continue
            if file_keyword not in file_name:
                continue
            print file_name
            file_count = file_count + 1
            train_time, test_time = results_from_file(folder_name+file_name, train_time_keyword,    test_time_keyword)
            if len(train_list) > fold_id:
                train_list[fold_id] = train_time
                test_list[fold_id] = test_time
            else:
                train_list.append(train_time)
                test_list.append(test_time)
    
    print np.average(train_list)
    print np.average(test_list)   


if __name__ == '__main__':
    data_key = "dsa"
    data_key = "rar"
    #data_key = "arc"
    #data_key = "ara"
    data_key = "asl"
    #data_key = "fixed_arc"
    method = "cnn"
    #method = "knn"
    #method = "libsvm"
    #method = "rf"

    if data_key == "dsa":
        top_k = "_top15_"
        num_classes = 19
    elif data_key == "rar":
        top_k = "_top30_"
        num_classes = 33
    elif data_key == "arc" or data_key == "fixed_arc":
        top_k = "_top30_"
        num_classes = 18
    elif data_key == "ara":
        top_k = "_top4_"
        num_classes = 10
    elif data_key == "asl":
        top_k = "_top6_"
        num_classes = 95

    log_folder = "../../log/" + data_key + "/"
    
    #folder_keyword = "multi_proj_feature_classification/cnn_obj_folder_rf_lda_sum"
    folder_keyword = "forward_wrapper"
    folder_keyword = "forward_wrapper_cnn"
    #folder_keyword = "cnn_classification"
    folder_name = log_folder + folder_keyword + "/"

    file_keyword = "top45"
    file_keyword = "top15"
    #file_keyword = "act3" 
    ###ASL
    #file_keyword = "top6"
    #file_keyword = "top22"

    ###RAR
    file_keyword = "top30"
    #file_keyword = "top117"
    file_keyword = ".log"


    #line_keyword = 'Top Features For Class '
    
    #acc_keyword = "Fold eval value"
    train_time_keyword = "fold training time"
    test_time_keyword = "fold testing time"
    #train_time_keyword = "Training time (sec)"
    #test_time_keyword = "Testing time (sec)"
    results_from_folder(folder_name, file_keyword, train_time_keyword, test_time_keyword)
    
