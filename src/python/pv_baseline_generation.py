# This python code is used to generate normal projected features
# 1. projected feature based on PCA
# 2. projected feature based on LDA

# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from os.path import isfile, join, isdir
from sklearn.decomposition import PCA
import time

from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from log_io import setup_logger
from data_io import file_reading
from data_io import x_y_spliting
from data_io import list_files
from data_io import init_folder
from object_io import save_obj
from parameter_proc import read_pure_feature_generation

from cnn_feature_evaluation import project_cnn_feature_combined_rf_lda_analysis
from cnn_feature_evaluation import project_cnn_feature_combined_rf_analysis
from cnn_feature_evaluation import project_cnn_feature_combined_lda_analysis
from cnn_feature_evaluation import map_attr_imp_analysis

# PCA analysis using sklearn pca package
# data_matrix: 2D matrix, N * C
# K: Top K PCA
# Return: N * K
def sklearn_pca_analysis(data_matrix, K):
    pca = PCA(n_components=K)
    pca.fit(data_matrix)
    #print pca.explained_variance_ratio_
    return pca.transform(data_matrix), None, pca.components_.T

# Do standardization
def standardization(data_matrix):
    data_matrix -= mean(data_matrix, 0)
    data_matrix /= std(data_matrix, 0)
    return data_matrix

def cov(data):
    """
        covariance matrix
        note: specifically for mean-centered data
    """
    N = data.shape[1]
    C = empty((N, N))
    for j in range(N):
        C[j, j] = mean(data[:, j] * data[:, j])
        for k in range(N):
            C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    return C


# This data_matrix, each column is one feature
def pca_source(data_matrix, K=None):
    """
        Principal component analysis using eigenvalues
        note: this mean-centers and auto-scales the data (in-place)
    """
    data_matrix = standardization(data_matrix)
    cov_matrix = cov(data_matrix)
    indices = np.where(np.isnan(cov_matrix))
    cov_matrix[indices]=0
    #print '++++'
    #print data_matrix
    #print cov_matrix
    #print '====='
#    eigh_vector, weight_matrix = eigh(cov_matrix)
    eigh_vector, weight_matrix = np.linalg.eigh(cov_matrix)
    key = argsort(eigh_vector)[::-1][:K]
    eigh_vector, weight_matrix = eigh_vector[key], weight_matrix[:, key]
    #pca_data_matrix = dot(weight_matrix.T, data_matrix.T).T
    pca_data_matrix = dot(data_matrix, weight_matrix)
    return pca_data_matrix, eigh_vector, weight_matrix


# the weight_matrix is V, with all_feature_num * k_component
# Get the absolute value of weight matrix, then get the sum for each row
# First return: the importance of each feature. 
# For example: if all_feature_num = 5, and first return = [4 1 3 0 2], it means the last feature is the most important one, then the second one, then the fourth one...
# Second return: The coresponding sum for each feature. 
def pca_model_analysis(weight_matrix):
    im_vector = np.sum(np.absolute(weight_matrix), axis=1)
    return argsort(im_vector)[::-1], im_vector


def run_pca_proj_feature_2D(data_matrix):
    row_num, col_num = data_matrix.shape
    pca_data_matrix, eigh_vector, weight_matrix = pca_source(data_matrix, col_num)
    im_vector_index, im_vector = pca_model_analysis(weight_matrix)
    return im_vector_index, im_vector


def run_pca_proj_feature_3D(d3_data_matrix):
    row_num, attr_num, attr_len = d3_data_matrix.shape
    all_im_vector = np.zeros(attr_num)
    for i in range(0, row_num):
        instance_matrix = d3_data_matrix[i, :, :].T
        #pca_data_matrix, eigh_vector, weight_matrix = pca_source(instance_matrix, attr_len)
        #im_vector_index, im_vector = pca_model_analysis(weight_matrix)
        pca_data_matrix, eigh_vector, weight_matrix = sklearn_pca_analysis(instance_matrix, attr_num)
        im_vector_index, im_vector = pca_model_analysis(weight_matrix)
        all_im_vector = all_im_vector + im_vector
    return argsort(all_im_vector)[::-1], all_im_vector


def run_lda_proj_feature_3D(x_matrix, y_vector, attr_num):
    row_num, col_num = x_matrix.shape
    attr_len = col_num/attr_num
    weight_vector = gene_lda_feature(x_matrix, y_vector)
    weight_vector = np.absolute(weight_vector).reshape(col_num)
    #print weight_vector.shape
    #print weight_vector

    ret_vector = np.zeros(attr_num)
    for i in range(0, col_num):
        attr_index = i/attr_len
        ret_vector[attr_index] = ret_vector[attr_index] + weight_vector[i]

    #print ret_vector
    return argsort(ret_vector)[::-1], ret_vector


def gene_projected_pca_feature(d3_data_matrix, y_vector, min_class, max_class, attr_num, transpose=False, logger=None):
    if logger == None:
        logger = init_logging('')
    pca_feature_array = []
    overall_time = 0

    for class_label in range(min_class, max_class+1):
        
        logger.info("calss label: " + str(class_label))
        #print "class: "+ str(class_label)
        class_index = np.where(y_vector==class_label)[0]
        class_data_matrix = d3_data_matrix[class_index, :, :]
        #print class_data_matrix.shape
        start_time = time.time()
        class_im_index, class_im_vector = run_pca_proj_feature_3D(class_data_matrix, )
        overall_time = overall_time + time.time() - start_time
        logger.info(class_im_index.shape)
        pca_feature_array.append(class_im_index)

    pca_feature_array = np.array(pca_feature_array)

    return pca_feature_array, overall_time





def run_pca_proj_feature_main(data_folder, class_column, attr_num, num_classes, pca_proj_obj_file, transpose=False, logger=None):
    if logger == None:
        logger = init_logging('')
    ret_pca_feature_array = []
    overall_time = 0

    file_list = listFiles(data_folder)

    file_count = 0
    for train_file in file_list:
        if "train" not in train_file: 
            continue
        logger.info(train_file)

        file_count = file_count + 1
        #if file_count > 2:
        #    break
        pca_feature_array = []

        x_matrix, y_vector = readFile(data_folder + train_file, class_column)
        row_num, col_num = x_matrix.shape
        attr_len = col_num/attr_num
        y_vector = y_vector.astype(int)
        start_class = min(y_vector)
        d3_data_matrix = x_matrix.reshape(row_num, attr_num, attr_len)

        for i in range(0, num_classes):
            class_label = i + start_class
            logger.info("calss label: " + str(i))
            #print "class: "+ str(class_label)
            class_index = np.where(y_vector==class_label)[0]
            class_data_matrix = d3_data_matrix[class_index, :, :]
            #print class_data_matrix.shape
            start_time = time.time()
            class_im_index, class_im_vector = run_pca_proj_feature_3D(class_data_matrix, )
            overall_time = overall_time + time.time() - start_time
            logger.info(class_im_index.shape)
            pca_feature_array.append(class_im_index)
        pca_feature_array = np.array(pca_feature_array)
        #print pca_feature_array.shape
        logger.info(pca_feature_array.shape)
        logger.info("end of " + train_file)
        ret_pca_feature_array.append(pca_feature_array)


    logger.info("Final:")
    ret_pca_feature_array = np.array(ret_pca_feature_array)
    logger.info(ret_pca_feature_array.shape)
    start_time = time.time()
    feature_array = fold_feature_combination_F_C_A(ret_pca_feature_array)
    overall_time = overall_time + time.time() - start_time
    #print feature_array.shape
    logger.info(feature_array.shape)
    logger.info(feature_array[0:3, 0:5])
    logger.info("Object saved to "+pca_proj_obj_file)
    logger.info("Overall time (sec): ")
    logger.info(str(overall_time))
    #print feature_array
    save_obj([feature_array], pca_proj_obj_file)

    return pca_feature_array


# Used to calculate the pure lda feature without CNN
def run_lda_proj_feature_main(data_folder, class_column, attr_num, num_classes, lda_proj_obj_file, transpose=False, logger=None):
    if logger == None:
        logger = init_logging('')

    file_list = listFiles(data_folder)
    overall_time = 0

    ret_lda_feature_array = []
    ret_lda_feature_weight = []

    file_count = 0
    lda_time = 0
    norm_time = 0
    for train_file in file_list:
        if "train" not in train_file: 
            continue
        logger.info(train_file)

        file_count = file_count + 1
        
        lda_feature_array = []

        x_matrix, y_vector = readFile(data_folder + train_file, class_column)
        #x_matrix = x_matrix[0:100, :]
        #y_vector = y_vector[0:100]

        row_num, col_num = x_matrix.shape
        logger.info(x_matrix.shape)
        attr_len = col_num/attr_num
        if transpose == True:
            x_matrix_transpose = []
            x_matrix = x_matrix.reshape(row_num, attr_num, attr_len)
            for r in range(4, attr_num):
                temp_x_matrix = x_matrix[:, r, :]
                fold_feature_matrix, fold_norm_time, fold_lda_time = gene_projected_lda_feature(temp_x_matrix, y_vector)
                print fold_feature_matrix
                sdfsd
                break
        else:
            start_time = time.time()
            fold_feature_matrix, fold_norm_time, fold_lda_time = gene_projected_lda_feature(x_matrix, y_vector)
            overall_time = overall_time + time.time() - start_time

        logger.info("fold norm: " + str(fold_norm_time))
        logger.info("fold lda: " + str(fold_lda_time))
        norm_time = fold_norm_time + norm_time
        lda_time = fold_lda_time + lda_time

        f_row_num, f_col_num = fold_feature_matrix.shape
        fold_feature_array = np.zeros((f_row_num, attr_num))
        fold_feature_weight_array = np.zeros((f_row_num, attr_num))
        logger.info(fold_feature_array.shape)
        for i in range(0, f_row_num):
            temp_vector = np.zeros(attr_num)
            for j in range(0, f_col_num):
                attr_index = j/attr_len
                temp_vector[attr_index] = temp_vector[attr_index] + fold_feature_matrix[i, j]
            
            fold_feature_weight_array[i, :] = temp_vector
            fold_feature_array[i, :] = argsort(temp_vector)[::-1]
 
        ret_lda_feature_array.append(fold_feature_array)
        ret_lda_feature_weight.append(fold_feature_weight_array)

    logger.info("overall norm: " + str(norm_time))
    logger.info("overall lda: " + str(lda_time))
    ret_lda_feature_weight = np.array(ret_lda_feature_weight)
    ret_lda_feature_array = np.array(ret_lda_feature_array)
    logger.info(ret_lda_feature_array.shape)

    ret_lda_feature_weight = np.sum(ret_lda_feature_weight, axis=0)
    
    ret_lda_feature_array = ret_lda_feature_array.astype(int)
    combine_time = 0
    start_time = time.time()
    lda_feature_array = fold_feature_combination_F_C_A(ret_lda_feature_array)
    combine_time = time.time() - start_time
    overall_time = overall_time + combine_time

    logger.info("combine lda: " + str(overall_time))
    logger.info(lda_feature_array.shape)
    logger.info(lda_feature_array[0:7, 0:7])
    logger.info(ret_lda_feature_weight[0:7, 0:7])
    logger.info("pure lda projected feature generation overall time (sec)")
    logger.info(overall_time)
    save_obj([lda_feature_array, ret_lda_feature_weight], lda_proj_obj_file)
    logger.info("Object saved to " + lda_proj_obj_file)

    return lda_feature_array




# Used to calculate the pure feature without CNN
def run_pure_proj_feature_main(file_keyword, parameter_file = '../../parameters/feature_generation_projected_pure.txt'):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, method, log_folder, out_obj_folder = read_pure_feature_generation(parameter_file)

    print data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, method, log_folder, out_obj_folder

    file_list = list_files(data_folder)
    overall_time = 0

    ret_feature_array = []
    ret_feature_weight = []

    file_count = 0
    overall_time = 0
    predict = False
    delimiter = " "
    header = True
    for train_file in file_list:
        if file_keyword not in train_file: 
            continue
        train_key = train_file.replace('.txt', '')
        file_count = file_count + 1
        
        data_matrix, attr_num = file_reading(data_folder + train_file)
        train_x_matrix, train_y_vector = x_y_spliting(data_matrix, class_column)
        train_row, train_col = train_x_matrix.shape
        train_x_matrix = train_x_matrix.reshape(train_row, attr_num, attr_len)
        if class_id < 0:
            min_class = min(train_y_vector)
            max_class = max(train_y_vector) + 1
        else:
            min_class = class_id
            max_class = min_class + 1
        log_file = train_key + "_" + method + "_min" + str(min_class) + "_max" + str(max_class) + "_pure_projected.log"

        #logger = setup_logger('')
        logger = setup_logger(log_folder + log_file)
        print log_folder + log_file
        logger.info(train_file)
        out_obj_file = train_key + "_" + method + "_min" + str(min_class) + "_max" + str(max_class) + "_pure_projected.obj"
        out_obj_matrix = []
        logger.info("min class: " + str(min_class))
        logger.info("max class: " + str(max_class))
        for label in range(min_class, max_class):
            class_train_y = np.where(train_y_vector == label, 1, 0)
            logger.info("label: " + str(label))
            if method == 'rf_lda':
                class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_rf_lda_analysis(train_x_matrix, class_train_y, logger)
            elif method == "rf":
                class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_rf_analysis(train_x_matrix, class_train_y, logger)
            elif method == "lda":
                class_attr_imp_matrix, class_run_time = project_cnn_feature_combined_lda_analysis(train_x_matrix, class_train_y, logger)
            logger.info("class attr imp matrix shape: " + str(class_attr_imp_matrix.shape))
            class_attr_list = map_attr_imp_analysis(class_attr_imp_matrix, logger)
            logger.info(class_attr_list)
            logger.info(class_attr_list.shape)
            out_obj_matrix.append(class_attr_list)

        out_obj_matrix = np.array(out_obj_matrix)
        logger.info("out obj to: " + out_obj_folder + out_obj_file)
        logger.info(out_obj_matrix.shape)
        save_obj([out_obj_matrix], out_obj_folder + out_obj_file)


if __name__ == '__main__':
    argv_array = sys.argv
    #data_matrix = np.random.rand(500, 10, 20)
    #index_vector, value_vector = run_pca_proj_feature_3D(data_matrix)
    #print index_vector
    #print value_vector
#
    #print index_vector.shape
    #print value_vector.shape
    #sdf
###
###
###    data_folder = "../../data/human/processed/ready/data.txt_trainTest10/"
###    class_column = 0
###    attr_num = 117
###    num_classes = 33
###    data_keyword = 'rar'
###
###    data_folder = "../../data/uci_human_movement/"
###    class_column = 0
###    attr_num = 45
###    num_classes = 19
###    data_keyword = 'uci'
###
###    data_folder = "../../data/arc_activity_recognition/s1_ijcal_10_folds/arc/all.txt_trainTest10/"
###    class_column = 0
###    attr_num = 107
###    num_classes = 18
###    data_keyword = 'arc'
###
###    data_folder = "../../data/evn/ds/DS_all_ready_to_model.csv_trainTest2_weekly_5attr/"
###    class_column = 0
###    attr_num = 5
###    num_classes = 2
###    data_keyword = 'evn/ds'
###
###   #data_folder = "../../data/evn/es/ES_all_ready_to_model.csv_trainTest2_weekly_5attr/"
###   #class_column = 0
###   #attr_num = 5
###   #num_classes = 2
###   #data_keyword = 'evn/es'
###
###   #data_folder = "../../data/evn/pg/PG_all_ready_to_model.csv_trainTest2_weekly_5attr/"
###   #class_column = 0
###   #attr_num = 5
###   #num_classes = 2
###   #data_keyword = 'evn/pg'
###
###
###    transpose = True
###
###
###    method = "pure_lda_projected_feature"
###    #method = "pure_pca_projected_feature"


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
            print ("That's not an int!")
    print file_keyword
    run_pure_proj_feature_main(file_keyword)

