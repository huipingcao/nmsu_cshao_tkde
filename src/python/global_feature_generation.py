# This python code is used to generate global features
# 1. global feature based on PCA
# 2. global feature based on LDA

# -*- coding: utf-8 -*-
import numpy as np
import sys
from sklearn.decomposition import PCA
import time

from numpy import dot, mean, std, empty, argsort
from numpy.linalg import eigh
from data_io import listFiles
from data_io import readFile
from data_io import init_logging

from parameter_proc import read_global_feature_generation_parameter
from object_io import save_obj
from object_io import load_obj

from feature_evaluation import gene_lda_feature_v2
from feature_evaluation import majority_vote
from post_classification import majority_vote_index


# PCA analysis using sklearn pca package
# data_matrix: 2D matrix, N * C
# K: Top K PCA
# Return: N * K
def sklearn_pca_analysis(data_matrix, K):
    pca = PCA(n_components=K)
    pca.fit(data_matrix)
    return pca.transform(data_matrix), pca.components_.T

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
    eigh_vector, weight_matrix = eigh(cov_matrix)
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


def run_pca_feature_2D(data_matrix):
    row_num, col_num = data_matrix.shape
    #pca_data_matrix, eigh_vector, weight_matrix = pca_source(data_matrix, col_num)
    pca_data_matrix, weight_matrix = sklearn_pca_analysis(data_matrix, col_num)
    im_vector_index, im_vector = pca_model_analysis(weight_matrix)
    return im_vector_index, im_vector



def gene_global_pca_feature(x_matrix, attr_num, logger):
    run_time = 0
    row_num, col_num = x_matrix.shape
    start_time = time.time()
    pca_vector_index, pca_vector = run_pca_feature_2D(x_matrix)
    run_time = run_time + time.time() - start_time

    logger.info("pca importance matrix shape: " + str(pca_vector.shape))
    attr_len = col_num/attr_num
    start_time = time.time()
    pca_vector = np.absolute(pca_vector)
    pca_vector = pca_vector.reshape(attr_num, attr_len)
    pca_vector = pca_vector.sum(axis=1)
    pca_vector_index = argsort(pca_vector)[::-1]
    run_time = run_time + time.time() - start_time
    logger.info("pca importance matrix shape: " + str(pca_vector.shape))
    return pca_vector_index, run_time


# Generate global lda feature based on the input data matrix and class label vector
# x_matrix: 2D matrix with I * (A * L): I is the number of instances, A is number of attribute and L is attribute length
# y_vector: vector with length I, it is the class labels for all instances
# attr_num: number of attributes
# return a vector with length attr_num. For example [2, 1, 0, 3], it means the the second (starts from 0) attribute is the most important one
def gene_global_lda_feature(x_matrix, y_vector, attr_num, logger):
    run_time = 0
    start_time = time.time()
    #print "=====x_matrix========"
    #print x_matrix.shape
    #print "======WRONG======="
    lda_coef_matrix = gene_lda_feature_v2(x_matrix, y_vector)
    run_time = run_time + time.time() - start_time
    logger.info("lda coefficient matrix shape: " + str(lda_coef_matrix.shape))
    coef_row, coef_col = lda_coef_matrix.shape
    attr_len = coef_col/attr_num
    start_time = time.time()
    lda_coef_matrix = np.absolute(lda_coef_matrix)
    lda_coef_matrix = lda_coef_matrix.reshape(coef_row, attr_num, attr_len)
    lda_coef_matrix = lda_coef_matrix.sum(axis=2)
    feature_index_vector, feature_coeff_vector = majority_vote(lda_coef_matrix, -1, True)
    run_time = run_time + time.time() - start_time
    logger.info("lda coefficient matrix shape: " + str(feature_index_vector.shape))
    return feature_index_vector, run_time



# Generate global lda features through all data files under data folder
def global_lda_pca_feature_main(parameter_file, method):
    data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, pckl_folder, log_folder, log_postfix, out_obj_folder = read_global_feature_generation_parameter(parameter_file)
   
    log_file = log_folder + data_keyword+'_'+method + log_postfix
    #log_file = '' # without write to file
    logger = init_logging(log_file)
    logger.info('METHOD: ' + method)
    logger.info('DATA KEYWORD: '+ data_keyword)
    logger.info('ATTRIBUTE NUMBER: '+ str(attr_num))
    logger.info('ATTRIBUTE LENGTH: '+ str(attr_len))
    logger.info('CLASS NUMBER: '+ str(num_classes))
    logger.info('CLASS COLUMN: '+ str(class_column))
    logger.info('START CLASS: '+ str(start_class))
    logger.info('PCKL FOLDER: '+ pckl_folder)
    logger.info('LOG FOLDER: '+ log_folder)
    logger.info('LOG POSTFIX: '+ log_postfix)
    logger.info('OUTPUT FOLDER: '+ out_obj_folder)


    function_name = sys._getframe(1).f_code.co_name
    logger = init_logging(log_file)

    file_list = listFiles(data_folder)
    overall_time = 0

    ret_feature_array = []

    file_count = 0
    #method = 'pca'
    for train_file in file_list:
        if "train" not in train_file: 
            continue 

        logger.info(train_file)
        
        lda_feature_array = []
        
        x_matrix, y_vector = readFile(data_folder + train_file, class_column)


        if file_count == 0:
            logger.info("x data matrix shape: "+str(x_matrix.shape))
            logger.info("y vector shape: " + str(y_vector.shape))

        row_num, col_num = x_matrix.shape
        attr_len = col_num/attr_num
        if method == 'lda':
            feature_index_vector, run_time = gene_global_lda_feature(x_matrix, y_vector, attr_num, logger)
        elif method == 'pca':
            feature_index_vector, run_time = gene_global_pca_feature(x_matrix, attr_num, logger)
        overall_time = overall_time + run_time
        ret_feature_array.append(feature_index_vector)

        file_count = file_count + 1
        #break
        #if file_count > 1:
        #    break

    ret_feature_array = np.matrix(ret_feature_array)
    logger.info(ret_feature_array.shape)
    logger.info("ret_feature_array samples:")
    logger.info("\n" + str(ret_feature_array[0:4, :]))

    start_time = time.time()
    ret_feature_index, ret_feature_value = majority_vote_index(ret_feature_array, -1)
    overall_time = overall_time + time.time() - start_time
    logger.info("\n" + str(ret_feature_index[0:6]))
    logger.info("global feature run time (sec): "+ str(overall_time))
    obj_file = out_obj_folder + '_' + method + "_global_feature.pckl"
    logger.info("global feature saved to: "+ str(obj_file))
    save_obj(ret_feature_index, obj_file)
    return ret_feature_index, overall_time
    

# Generate global pca features through all data files under data folder
def global_cnn_pca_feature_main(parameter_file, method):
    data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, pckl_folder, log_folder, log_postfix, out_obj_folder = read_global_feature_generation_parameter(parameter_file)
   
    log_file = log_folder + data_keyword+'_'+method + log_postfix
    #log_file = '' # without write to file
    logger = init_logging(log_file)
    logger.info('METHOD: ' + method)
    logger.info('DATA KEYWORD: '+ data_keyword)
    logger.info('ATTRIBUTE NUMBER: '+ str(attr_num))
    logger.info('ATTRIBUTE LENGTH: '+ str(attr_len))
    logger.info('CLASS NUMBER: '+ str(num_classes))
    logger.info('CLASS COLUMN: '+ str(class_column))
    logger.info('START CLASS: '+ str(start_class))
    logger.info('PCKL FOLDER: '+ pckl_folder)
    logger.info('LOG FOLDER: '+ log_folder)
    logger.info('LOG POSTFIX: '+ log_postfix)
    logger.info('OUTPUT FOLDER: '+ out_obj_folder)


    function_name = sys._getframe(1).f_code.co_name

    file_list = listFiles(pckl_folder)
    overall_time = 0

    ret_feature_array = []

    file_count = 0
    for train_file_pckl in file_list:
        if "train" not in train_file_pckl: 
            continue 

        logger.info(train_file_pckl)
        out_matrix, weight_matrix, bias_vector = load_obj(pckl_folder + train_file_pckl)
        out_matrix = np.squeeze(out_matrix)
        
        if file_count == 0:
            logger.info('layer out matrix shape: ' + str(out_matrix.shape))
            logger.info('weight matrix shape: ' + str(weight_matrix.shape))
            logger.info('bias vector shape: ' + str(bias_vector.shape))

        row_num, attr_num, attr_len = out_matrix.shape
        logger.info('out matrix reshape: ' + str(out_matrix.shape))
        out_matrix = out_matrix.reshape(row_num, attr_num*attr_len)
        logger.info(out_matrix.shape)
        feature_index_vector, run_time = gene_global_pca_feature(out_matrix, attr_num, logger)
        overall_time = overall_time + run_time
        logger.info(feature_index_vector.shape)
        ret_feature_array.append(feature_index_vector)

        file_count = file_count + 1
        #break
        #if file_count > 1:
        #    break

    ret_feature_array = np.matrix(ret_feature_array)
    logger.info(ret_feature_array.shape)
    logger.info("return feature array samples:")
    logger.info("\n" + str(ret_feature_array[0:4, 0:6]))

    start_time = time.time()
    ret_feature_index, ret_feature_value = majority_vote_index(ret_feature_array, -1)
    overall_time = overall_time + time.time() - start_time
    logger.info("\n" + str(ret_feature_index[0:6]))
    logger.info(method + " global feature run time (sec): "+ str(overall_time))
    obj_file = out_obj_folder + method + "_global_feature.pckl"
    logger.info('output feature object saved: '+obj_file)
    save_obj(ret_feature_index, obj_file)
    return ret_feature_index, overall_time




# Generate global lda features through all data files under data folder
def global_cnn_lda_feature_main(parameter_file, method):
    data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, pckl_folder, log_folder, log_postfix, out_obj_folder = read_global_feature_generation_parameter(parameter_file)
   
    log_file = log_folder + data_keyword+'_'+method + log_postfix
    #log_file = '' # without write to file
    logger = init_logging(log_file)
    logger.info('METHOD: ' + method)
    logger.info('DATA KEYWORD: '+ data_keyword)
    logger.info('ATTRIBUTE NUMBER: '+ str(attr_num))
    logger.info('ATTRIBUTE LENGTH: '+ str(attr_len))
    logger.info('CLASS NUMBER: '+ str(num_classes))
    logger.info('CLASS COLUMN: '+ str(class_column))
    logger.info('START CLASS: '+ str(start_class))
    logger.info('PCKL FOLDER: '+ pckl_folder)
    logger.info('LOG FOLDER: '+ log_folder)
    logger.info('LOG POSTFIX: '+ log_postfix)
    logger.info('OUTPUT FOLDER: '+ out_obj_folder)

    function_name = sys._getframe(1).f_code.co_name
    logger = init_logging(log_file)

    file_list = listFiles(pckl_folder)
    overall_time = 0

    ret_feature_array = []
    file_count = 0
    for train_file_pckl in file_list:
        if "train" not in train_file_pckl: 
            continue 
        train_file = train_file_pckl[0:train_file_pckl.index('.txt')] + '.txt'
        logger.info("PCKL FILE: " + train_file_pckl)
        logger.info("DATA FILE: "+ train_file)
        
        train_x_matrix, train_y_vector = readFile(data_folder + train_file)

        out_matrix, weight_matrix, bias_vector = load_obj(pckl_folder + train_file_pckl)
        out_matrix = np.squeeze(out_matrix)
        row_num, attr_num, attr_len = out_matrix.shape
        out_matrix = out_matrix.reshape(row_num, attr_num*attr_len)
        if file_count == 0:
            logger.info('layer out matrix shape: ' + str(out_matrix.shape))
            logger.info('weight matrix shape: ' + str(weight_matrix.shape))
            logger.info('bias vector shape: ' + str(bias_vector.shape))

        feature_index_vector, run_time = gene_global_lda_feature(out_matrix, train_y_vector, attr_num, logger)
        overall_time = overall_time + run_time
        logger.info(feature_index_vector.shape)
        ret_feature_array.append(feature_index_vector)

        file_count = file_count + 1
        break
        #if file_count > 1:
        #    break

    ret_feature_array = np.matrix(ret_feature_array)
    logger.info(ret_feature_array.shape)
    logger.info("return feature array samples:")
    logger.info("\n" + str(ret_feature_array[0:4, 0:6]))

    start_time = time.time()
    ret_feature_index, ret_feature_value = majority_vote_index(ret_feature_array, -1)
    overall_time = overall_time + time.time() - start_time
    logger.info("\n" + str(ret_feature_index[0:6]))
    logger.info(method + " global feature run time (sec): "+ str(overall_time))
    obj_file = out_obj_folder + method + "_global_feature.pckl"
    save_obj(ret_feature_index, obj_file)
    return ret_feature_index, overall_time
    



if __name__ == '__main__':
    argv_array = sys.argv
    log_stdout = sys.stdout


    print "USAGE: python global_feature_generation.py  <Input_Method>"
    print "<Input_Method> can be one of [cnn_lda, cnn_pca, lda, pca]"

    global_feature_generation_file = '../../parameters/global_feature_generation.txt'

    #method, pckl_folder, log_folder, log_postfix, out_obj_folder = read_global_feature_generation_parameter(global_feature_generation_file)
    #->Here I add the args[0] = method
    if len(sys.argv) == 2:
        method = sys.argv[1]
    else:
        method = 'lda'
        #raise Exception("USAGE: python global_feature_generation.py  <Input_Method> ")
    #The method we can choose is cnn_lda,cnn_pca,lda and pca
    
       
    print "METHOD: " + method    

    if method == 'cnn_lda':
        global_cnn_lda_feature_main(global_feature_generation_file, method)
    elif method == 'cnn_pca':
        global_cnn_pca_feature_main(global_feature_generation_file, method)
    elif method == 'lda' or method == 'pca':
        ret_feature_index, overall_time = global_lda_pca_feature_main(global_feature_generation_file, method)
