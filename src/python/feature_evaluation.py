import numpy as np
import sys
import time
from log_io import init_logging
from sklearn import preprocessing

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classification_results import averaged_class_based_accuracy
from classification_results import f1_value_precision_recall_accuracy


# x_matrix: 2D matrix, num_instance * attr_num
# y_vector: 1D vector, num_instance
def rf_feature_extraction(x_matrix, y_vector, predict=False, logger=None, rf_estimator=50):
    if logger is None:
        logger = init_logging('')

    rf_model = ExtraTreesClassifier(n_estimators=rf_estimator, random_state=0)
    start_time = time.time()
    rf_model.fit(x_matrix, y_vector)
    run_time = time.time() - start_time

    feature_value_vector = np.absolute(rf_model.feature_importances_)
    #sum_value = float(np.sum(feature_value_vector))
    #feature_value_vector = feature_value_vector/sum_value
    #feature_value_vector = preprocessing.normalize(feature_value_vector.reshape(1, len(feature_value_vector)), norm='l2')[0]
    if predict is True:
        predict_y = rf_model.predict(x_matrix)
        #averaged_acc = averaged_class_based_accuracy(predict_y, y_vector)
        accuracy, precision, recall, f1_value, tp, fp, tn, fn = f1_value_precision_recall_accuracy(predict_y, y_vector, 1)
    else:
        averaged_acc = -1
    return feature_value_vector, rf_model, f1_value, run_time


# x_matrix: 2D matrix, num_instance * attr_num
# y_vector: 1D vector, num_instance
def lda_feature_extraction(x_matrix, y_vector, predict=False, logger=None):
    if logger is None:
        logger = init_logging('')
    train_norm_vector = np.linalg.norm(x_matrix, axis=0, ord=np.inf)[None, :]
    #print "train_norm_vector"
    #print train_norm_vector
    x_matrix = np.true_divide(x_matrix, train_norm_vector, where=(train_norm_vector!=0))
    x_matrix[np.isnan(x_matrix)] = 0
    x_matrix[np.isinf(x_matrix)] = 1
    if x_matrix.max() == x_matrix.min():
        return None, None, -1, 0
    if np.any(x_matrix) is False:
        return None, None, -1, 0
    prior_vector = []
    min_class = min(y_vector)
    max_class = max(y_vector) + 1
    all_count = len(y_vector)
    for i in range(min_class, max_class):
        c_count = len(np.where(y_vector==i)[0])
        prior_vector.append(float(c_count)/all_count)
    lda_model = LinearDiscriminantAnalysis(priors=prior_vector)
    start_time = time.time()
    lda_model.fit(x_matrix, y_vector)
    run_time = time.time() - start_time
    feature_value_vector = np.absolute(lda_model.scalings_.T[0])
    #sum_value = float(np.sum(feature_value_vector))
    #feature_value_vector = feature_value_vector/sum_value
    #feature_value_vector = preprocessing.normalize(feature_value_vector.reshape(1, len(feature_value_vector)), norm='l2')[0]
    if predict is True:
        predict_y = lda_model.predict(x_matrix)
        #averaged_acc = averaged_class_based_accuracy(predict_y, y_vector)
        accuracy, precision, recall, f1_value, tp, fp, tn, fn = f1_value_precision_recall_accuracy(predict_y, y_vector, 1)
    else:
        averaged_acc = -1

    return feature_value_vector, lda_model, f1_value, run_time
    #return feature_value_vector, feature_value_index, lda_model




# feature_matrix: 3D matrix, num_instance * attr_num * num_map
# y_vector: 1D vector, num_instance
def cnn_feature_rf_analysis_main(feature_matrix, y_vector, logger):
    num_instance, num_attribute, num_map = feature_matrix.shape
    map_attr_imp_matrix = []  # used to store all attribute importance from each map
    map_attr_imp_index_matrix = []  # used to store
    overall_time = 0
    for i in range(0, num_map):
        map_feature_matrix = feature_matrix[:, :, i]
        start_time = time.time()
        feature_value_vector, averaged_acc = cnn_feature_rf(map_feature_matrix, y_vector)
        overall_time = overall_time + time.time() - start_time
        if i == 0:
            logger.info(feature_value_vector.shape)
        map_attr_imp_matrix.append(feature_value_vector)

    map_attr_imp_matrix = np.array(map_attr_imp_matrix)
    logger.info(map_attr_imp_matrix.shape)
    return map_attr_imp_matrix, overall_time


# feature_matrix: 3D matrix, num_instance * attr_num * num_map
# y_vector: 1D vector, num_instance
def cnn_feature_rf_analysis_main_v1(feature_matrix, y_vector, logger):
    num_instance, num_attribute, num_map = feature_matrix.shape
    map_attr_imp_matrix = [] # used to store all attribute importance from each map
    map_attr_imp_index_matrix = [] # used to store 
    overall_time = 0
    feature_matrix_2d = feature_matrix.reshape(num_instance, num_attribute*num_map)
    print "feature matrix shape"
    print feature_matrix_2d.shape
    print "==="
    feature_value_vector, averaged_acc = cnn_feature_rf(feature_matrix_2d, y_vector)
    feature_value_matrix = feature_value_vector.reshape(num_attribute, num_map)
    print "feature_value_matrix shape"
    print feature_value_matrix.shape
    print "==="
    feature_value_vector = np.sum(feature_value_matrix, axis=1)
    feature_value_vector = preprocessing.normalize(feature_value_vector.reshape(1, len(feature_value_vector)), norm='l2')[0]

    print "feature_value_vector shape"
    print feature_value_vector.shape
    print "==="

    return feature_value_vector, overall_time


def cnn_feature_lda(train_x_matrix, train_y_vector, predict=False, logger=None):
    if logger == None:
        logger = init_logging('')
    train_norm_vector = np.linalg.norm(train_x_matrix, axis=0, ord=np.inf)[None, :]
    #print train_norm_vector
    train_x_matrix = np.true_divide(train_x_matrix, train_norm_vector, where=(train_norm_vector!=0))
    train_x_matrix[np.isnan(train_x_matrix)] = 0
    train_x_matrix[np.isinf(train_x_matrix)] = 1
    #print train_x_matrix[0:3, 0:5]
    lda_model,train_time =  bi_gene_lda_model(train_x_matrix, train_y_vector)
    #feature_value_vector = np.absolute(lda_model.coef_[0])
    feature_value_vector = np.absolute(lda_model.scalings_.T[0])
    #logger.info("predict_bool: " + str(predict))
    if predict == True:
        feature_value_vector = preprocessing.normalize(feature_value_vector.reshape(1, len(feature_value_vector)), norm='l2')[0]
        predict_y = lda_model.predict(train_x_matrix)

        averaged_acc = averaged_class_based_accuracy(predict_y, train_y_vector)

        #mean_acc = lda_model.score(train_x_matrix, train_y_vector)
        #logger.info('mean_acc: ' + str(mean_acc))

        #logger.info('averaged_acc: ' + str(averaged_acc))
    else:
        averaged_acc = -1

    len_weight = len(feature_value_vector)
    sort_feature_value_vector = np.argsort(feature_value_vector)
    feature_vector_norm = np.zeros(len_weight)
    for i in range(0, len_weight):
        feature_vector_norm[sort_feature_value_vector[i]] = i

    return feature_vector_norm, feature_value_vector, lda_model, averaged_acc
    #return feature_value_vector, feature_value_index, lda_model

def project_cnn_feature_combined_rf_lda_analysis(feature_matrix, y_vector, logger=None):
    if logger==None:
        logger = init_logging('')

    num_instance, num_attribute, num_map = feature_matrix.shape
    map_attr_imp_matrix = [] # used to store all attribute importance from each map
    map_attr_imp_index_matrix = [] # used to store 
    predict = True
    skip_count = 0
    rf_time = 0
    lda_time = 0
    for i in range(0, num_map):
        map_feature_matrix = feature_matrix[:, :, i]
        start_time = 0
        feature_vector_norm, feature_value_vector, rf_model, averaged_acc = rf_feature_extraction(map_feature_matrix, y_vector, predict, logger)    
        rf_time = rf_time + time.time() - start_time
        #if averaged_acc != -1:
        #    logger.info("RF accuracy:")
        #    logger.info(averaged_acc)
        #    feature_value_vector = feature_value_vector * averaged_acc
        if i == 0:
            logger.info(feature_value_vector.shape)

        #if np.any(map_feature_matrix) == False:
            #print "do not know why"
        #    skip_count = skip_count + 1
        #    map_attr_imp_matrix.append(feature_value_vector)
        #    continue
        start_time = 0
        lda_feature_vector_norm, lda_feature_value_vector, lda_model, lda_averaged_acc = gene_lda_feature_v2(map_feature_matrix, y_vector, predict, logger)
        lda_time = lda_time + time.time() - start_time
        #if lda_averaged_acc != -1:
        #    logger.info("LDA accuracy:")
        #    logger.info(lda_averaged_acc)
        #    lda_feature_value_vector = lda_feature_value_vector * lda_averaged_acc
        feature_value_vector = feature_value_vector + lda_feature_value_vector
            
        #print feature_value_vector
        map_attr_imp_matrix.append(feature_value_vector)
        #len_feature = len(feature_value_vector)
        #sort_weight_vector = np.argsort(feature_value_vector)
        #feature_vector_norm = np.zeros(len_feature)
        #for i in range(0, len_feature):
        #    feature_vector_norm[sort_weight_vector[i]] = i
        #map_attr_imp_matrix.append(feature_vector_norm)

    map_attr_imp_matrix = np.array(map_attr_imp_matrix)
    logger.info(map_attr_imp_matrix.shape)

    return map_attr_imp_matrix, rf_time + lda_time



# For each feature_matrix from CNN, do analysis for this feature matrix
# feature_matrix: N * A * M: N: number of instances; A: number of attributes; M: number of maps
# For example: 8208 * 45 * 20 for UCI dataset. There are 8208 instances, 45 attribute, and the output map number is 20
# For each 8208 * 45 matrix, we need to analysis the importance for all 45 features and get a feature importance vector with length=45
# Then do majority vote on all 20 feature vectors
def project_cnn_feature_lda_analysis(feature_matrix, y_vector, logger):
    num_instance, num_attribute, num_map = feature_matrix.shape
    map_attr_imp_matrix = [] # used to store all attribute importance from each map
    map_attr_imp_index_matrix = [] # used to store 
    skip_count = 0
    overall_time = 0
    for i in range(0, num_map):
        map_feature_matrix = feature_matrix[:, :, i]
        
        if np.any(map_feature_matrix) == False:
            #print "do not know why"
            skip_count = skip_count + 1
            continue
        #print "Count number for neuron: " + str(i)
        #print "gene_projected_lda_feature:"
        #print map_feature_matrix.shape
        #print y_vector.shape
        start_time = time.time()
        feature_vector_norm, feature_value_vector, rf_model, averaged_acc =gene_lda_feature_v2(map_feature_matrix, y_vector)
        overall_time = overall_time + time.time() - start_time
        #feature_vector_norm, weight_vector_index, lda_model =gene_lda_feature_v2(map_feature_matrix, y_vector)
        if i == 0:
            logger.info(map_feature_matrix.shape)
        #print map_feature_matrix.shape
        #map_attr_imp_matrix.append(np.squeeze(map_feature_matrix))
        map_attr_imp_matrix.append(feature_vector_norm)
        #print map_feature_matrix
        #if i == 0:
        #    map_attr_imp_matrix = map_feature_matrix
        #else:
        #    map_attr_imp_matrix = map_feature_matrix.size
        #    map_attr_imp_matrix = np.add(map_attr_imp_matrix, map_feature_matrix)

    #print map_attr_imp_matrix.shape
    # we use negative map_attr_imp_matrix, the first number is the most important attribut number
    
    map_attr_imp_matrix = np.array(map_attr_imp_matrix)
    logger.info(map_attr_imp_matrix.shape)

    map_attr_imp_matrix = np.sum(map_attr_imp_matrix, axis=0)

    #map_attr_imp_index_matrix = argsort(-map_attr_imp_matrix, axis=1)
    
    #map_attr_imp_index_matrix, map_attr_imp_matrix = majority_vote(map_attr_imp_matrix, -1)

    return map_attr_imp_matrix, skip_count, overall_time




def run_cnn_combined_rf_lda_class_based_feature_analysis_main(feature_folder, feature_file_pre, feature_file_post, start_class, end_class, data_folder, data_stru, logger=None):
    if logger == None:
        logger = init_logging('')

    feature_weight_feature_matrix = []
    data_file_list = listFiles(data_folder)
    overall_time = 0
    for train_file in data_file_list:
        if 'train_' not in train_file:
            continue
        logger.info(train_file)
        train_keyword = train_file.replace('.txt', '')
        fold_feature_weight_matrix = []
        train_x_matrix, train_y_vector = readFile(data_folder + train_file)
        for class_label in range(start_class, end_class):
            logger.info("class label: " + str(class_label))
            feature_file = feature_file_pre + train_keyword + "_class_" + str(class_label) + feature_file_post
            [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + feature_file)
            fold_train_sensor_result = np.squeeze(fold_train_sensor_result)

            temp_train_y_vector = np.where(train_y_vector==class_label, 1, 0)
            fold_attr_imp_index, fold_time = project_cnn_feature_combined_rf_lda_analysis(fold_train_sensor_result, temp_train_y_vector, logger)
            overall_time = overall_time + fold_time

            if class_label == 0:
                logger.info(fold_train_sensor_result.shape)
                logger.info(fold_weight_fullconn.shape)
                logger.info(fold_bias_fullconn.shape)
                logger.info(fold_attr_imp_index.shape)
            fold_feature_weight_matrix.append(fold_attr_imp_index)

        fold_feature_weight_matrix = np.array(fold_feature_weight_matrix)
        logger.info("fold_feature_weight_matrix.shape")
        logger.info(fold_feature_weight_matrix.shape)

        fold_feature_weight_matrix = np.sum(fold_feature_weight_matrix, axis=1)

        logger.info("fold_feature_weight_matrix final shape")
        logger.info(fold_feature_weight_matrix.shape)

        feature_weight_feature_matrix.append(fold_feature_weight_matrix)

    feature_weight_feature_matrix = np.array(feature_weight_feature_matrix)
    logger.info("feature_weight_feature_matrix.shape")
    logger.info(feature_weight_feature_matrix.shape)

    start_time = time.time()
    feature_weight_feature_matrix = np.sum(feature_weight_feature_matrix, axis=0)
    feature_index_feature_matrix = np.argsort(-feature_weight_feature_matrix, axis=1)
    overall_time = overall_time + time.time() - start_time
    logger.info(feature_index_feature_matrix.shape)
    logger.info(feature_index_feature_matrix[0:5, 0:6])
    logger.info(feature_weight_feature_matrix.shape)
    logger.info(feature_weight_feature_matrix[0:5, 0:6])
    logger.info("fold cnn combined rf and lda projected feature generation overall time (sec)")
    logger.info(overall_time)
    return feature_index_feature_matrix, feature_weight_feature_matrix, overall_time


    ###feature_file_list = listFiles(feature_folder)
    ###overall_time = 0
    ###file_count = 0
    ###class_keyword = 'class_'+ str(class_label)+'_'
###
    ###file_demiliter = '_'
###
    ###for feature_file in feature_file_list:
    ###    if feature_file_keyword not in feature_file or class_keyword not in feature_file:
    ###        continue
    ###    logger.info(feature_file)
    ###    
    ###    file_count = file_count + 1
    ###    feature_file_array = feature_file.split(file_demiliter)
    ###    train_file = feature_file_array[1] + file_demiliter + feature_file_array[2] + '.txt'
    ###    logger.info(train_file)
    ###    train_x_matrix, train_y_vector = readFile(data_folder + train_file)
    ###    temp_train_y_vector = np.where(train_y_vector==class_label, 1, 0)
    ###    fold_positive_len = len(np.where(temp_train_y_vector == 1)[0])
    ###    fold_negative_len = len(temp_train_y_vector) - fold_positive_len
###
    ###    logger.info("=====")
    ###    logger.info("positive class labels length: " + str(fold_positive_len))
    ###    logger.info("negative class labels length: " + str(fold_negative_len))
###
    ###    [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + feature_file)
###
    ###    logger.info(fold_train_sensor_result.shape)
    ###    logger.info(fold_weight_fullconn.shape)
    ###    logger.info(fold_bias_fullconn.shape)
###
    ###    fold_train_sensor_result = np.squeeze(fold_train_sensor_result)
    ###    logger.info(fold_train_sensor_result.shape)
    ###    start_time = time.time()
###
###
    ###    fold_attr_imp_index = project_cnn_feature_combined_rf_lda_analysis(fold_train_sensor_result, temp_train_y_vector, logger)
    ###    overall_time = overall_time + time.time() - start_time
    ###    logger.info(fold_attr_imp_index.shape)
    ###    feature_weight_feature_matrix.append(fold_attr_imp_index)
    ###    #if file_count > 2:
    ###    #    break
        #break



    ###feature_weight_feature_matrix = np.array(feature_weight_feature_matrix)
    ###logger.info(feature_weight_feature_matrix.shape)
###
    ###start_time = time.time()
    ###class_based_value_matrix = np.sum(feature_weight_feature_matrix, axis=1)
    ###class_based_index_matrix = np.argsort(-class_based_value_matrix, axis=1)
    ###overall_time = overall_time + time.time() - start_time
    ###logger.info(class_based_index_matrix.shape)
    ###logger.info(class_based_index_matrix[0:5, 0:6])
    ###logger.info("fold cnn combined rf and lda projected feature generation overall time (sec)")
    ###logger.info(overall_time)
    ###return class_based_index_matrix, class_based_value_matrix, overall_time





def run_cnn_rf_class_based_feature_analysis_main(feature_folder, feature_file_keyword, class_label, data_folder, data_stru, logger=None):
    if logger == None:
        logger = init_logging('')

    feature_weight_feature_matrix = []

    feature_file_list = listFiles(feature_folder)
    overall_time = 0
    file_count = 0
    class_keyword = 'class_'+ str(class_label)+'_'

    file_demiliter = '_'

    for feature_file in feature_file_list:
        if feature_file_keyword not in feature_file or class_keyword not in feature_file:
            continue
        logger.info(feature_file)
        
        file_count = file_count + 1
        feature_file_array = feature_file.split(file_demiliter)
        train_file = feature_file_array[1] + file_demiliter + feature_file_array[2] + '.txt'
        logger.info(train_file)
        train_x_matrix, train_y_vector = readFile(data_folder + train_file)
        temp_train_y_vector = np.where(train_y_vector==class_label, 1, 0)
        fold_positive_len = len(np.where(temp_train_y_vector == 1)[0])
        fold_negative_len = len(temp_train_y_vector) - fold_positive_len

        logger.info("=====")
        logger.info("positive class labels length: " + str(fold_positive_len))
        logger.info("negative class labels length: " + str(fold_negative_len))

        [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + feature_file)

        logger.info(fold_train_sensor_result.shape)
        logger.info(fold_weight_fullconn.shape)
        logger.info(fold_bias_fullconn.shape)

        fold_train_sensor_result = np.squeeze(fold_train_sensor_result)
        logger.info(fold_train_sensor_result.shape)
        fold_attr_imp_index, fold_time = project_cnn_feature_rf_analysis(fold_train_sensor_result, temp_train_y_vector, logger)
        overall_time = overall_time + fold_time

        logger.info(fold_attr_imp_index.shape)
        feature_weight_feature_matrix.append(fold_attr_imp_index)
        #if file_count > 2:
        #    break
        #break



    feature_weight_feature_matrix = np.array(feature_weight_feature_matrix)
    logger.info(feature_weight_feature_matrix.shape)

    start_time = time.time()
    class_based_value_matrix = np.sum(feature_weight_feature_matrix, axis=1)
    class_based_index_matrix = np.argsort(-class_based_value_matrix, axis=1)
    overall_time = overall_time + time.time() - start_time
    logger.info(class_based_index_matrix.shape)
    logger.info(class_based_index_matrix[0:5, 0:6])
    logger.info("fold cnn lda projected feature generation overall time (sec)")
    logger.info(overall_time)
    return class_based_index_matrix, class_based_value_matrix, overall_time



def run_cnn_lda_feature_analysis_main(feature_folder, feature_file_keyword, data_folder, data_stru, feature_postfix, logger=None):
    if logger == None:
        logger = init_logging('')
    num_classes = data_stru.num_classes
    start_class = data_stru.start_class
    attr_num = data_stru.attr_num
    class_column = data_stru.class_column

    lda_weight_feature_matrix = []

    file_list = listFiles(data_folder)
    feature_file_list = listFiles(feature_folder)
    overall_time = 0
    file_count = 0
    file_demiliter = '_'
    for train_file in file_list:
        logger.info(train_file)
        file_count = file_count + 1
        for feature_file in feature_file_list:
            if feature_file_keyword not in feature_file or train_file not in feature_file: 
                continue
            logger.info(feature_file)
            train_x_matrix, train_y_vector = readFile(data_folder + train_file)
            [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + feature_file)
    
            logger.info(fold_train_sensor_result.shape)
            logger.info(fold_weight_fullconn.shape)
            logger.info(fold_bias_fullconn.shape)
    
            fold_train_sensor_result = np.squeeze(fold_train_sensor_result)
            logger.info(fold_train_sensor_result.shape)
            fold_attr_imp_index, fold_attr_imp, skip_count, fold_time = project_cnn_feature_lda_analysis(fold_train_sensor_result, train_y_vector, logger)
            overall_time = overall_time + fold_time
            logger.info(fold_attr_imp_index.shape)
            logger.info(fold_attr_imp.shape)
            logger.info("skip: " + str(skip_count))
            lda_weight_feature_matrix.append(fold_attr_imp_index)

            #if file_count > 2:
            #    break
            #break
    lda_weight_feature_matrix = np.array(lda_weight_feature_matrix)
    lda_weight_feature_matrix = lda_weight_feature_matrix.astype(int)
    logger.info(lda_weight_feature_matrix.shape)
    start_time = time.time()
    lda_feature_matrix = fold_feature_combination_F_C_A(lda_weight_feature_matrix)
    logger.info(lda_feature_matrix.shape)
    overall_time = overall_time + time.time() - start_time
    logger.info(lda_feature_matrix.shape)
    logger.info(lda_feature_matrix[0:5, 0:6])
    return lda_feature_matrix, overall_time
    #save_obj([lda_feature_matrix], lda_proj_obj_file)



def run_cnn_lda_class_based_feature_analysis_main(feature_folder, feature_file_keyword, class_label, data_folder, data_stru, logger=None):
    if logger == None:
        logger = init_logging('')

    lda_weight_feature_matrix = []

    feature_file_list = listFiles(feature_folder)
    overall_time = 0
    file_count = 0
    class_keyword = 'class_'+ str(class_label)+'_'

    file_demiliter = '_'

    for feature_file in feature_file_list:
        if feature_file_keyword not in feature_file or class_keyword not in feature_file:
            continue
        logger.info(feature_file)
        
        file_count = file_count + 1
        feature_file_array = feature_file.split(file_demiliter)
        train_file = feature_file_array[1] + file_demiliter + feature_file_array[2] + '.txt'
        logger.info(train_file)
        train_x_matrix, train_y_vector = readFile(data_folder + train_file)
        temp_train_y_vector = np.where(train_y_vector==class_label, 1, 0)
        fold_positive_len = len(np.where(temp_train_y_vector == 1)[0])
        fold_negative_len = len(temp_train_y_vector) - fold_positive_len

        logger.info("=====")
        logger.info("positive class labels length: " + str(fold_positive_len))
        logger.info("negative class labels length: " + str(fold_negative_len))

        [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + feature_file)

        logger.info(fold_train_sensor_result.shape)
        logger.info(fold_weight_fullconn.shape)
        logger.info(fold_bias_fullconn.shape)

        fold_train_sensor_result = np.squeeze(fold_train_sensor_result)
        logger.info(fold_train_sensor_result.shape)
        fold_attr_imp_index, skip_count, fold_time = project_cnn_feature_lda_analysis(fold_train_sensor_result, temp_train_y_vector, logger)
        overall_time = overall_time + fold_time
        logger.info(fold_attr_imp_index.shape)
        logger.info("skip: " + str(skip_count))
        lda_weight_feature_matrix.append(fold_attr_imp_index)
        #if file_count > 2:
        #    break
        #break



    lda_weight_feature_matrix = np.array(lda_weight_feature_matrix)
    lda_weight_feature_matrix = np.squeeze(lda_weight_feature_matrix)
    logger.info(lda_weight_feature_matrix.shape)

    start_time = time.time()
    class_based_value_array = np.sum(lda_weight_feature_matrix, axis=0)
    class_based_index_array = np.argsort(-class_based_value_array)
    overall_time = overall_time + time.time() - start_time
    return class_based_index_array, class_based_value_array, overall_time




def run_cnn_pca_class_based_feature_analysis_main(feature_folder, feature_file_keyword, class_label, data_folder, data_stru, logger=None):
    if logger == None:
        logger = init_logging('')

    weight_feature_matrix = []

    feature_file_list = listFiles(feature_folder)
    overall_time = 0
    file_count = 0
    class_keyword = 'class_'+ str(class_label)+'_'

    file_demiliter = '_'

    for feature_file in feature_file_list:
        if feature_file_keyword not in feature_file or class_keyword not in feature_file:
            continue
        logger.info(feature_file)
        #print(feature_file)
        file_count = file_count + 1
        feature_file_array = feature_file.split(file_demiliter)
        train_file = feature_file_array[1] + file_demiliter + feature_file_array[2] + '.txt'
        logger.info(train_file)
        train_x_matrix, train_y_vector = readFile(data_folder + train_file)

        class_label_index = np.where(train_y_vector==class_label)[0]

        logger.info("=====")
        logger.info("positive class labels length: " + str(len(class_label_index)))

        [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + feature_file)

        logger.info(fold_train_sensor_result.shape)
        logger.info(fold_weight_fullconn.shape)
        logger.info(fold_bias_fullconn.shape)

        fold_train_sensor_result = np.squeeze(fold_train_sensor_result)
        fold_train_sensor_result = fold_train_sensor_result[class_label_index, :, :]
        logger.info(fold_train_sensor_result.shape)
        start_time = time.time()
        fold_attr_imp_index, fold_attr_imp = run_pca_proj_feature_3D(fold_train_sensor_result)
        overall_time = overall_time + time.time() - start_time
        logger.info(fold_attr_imp_index.shape)
        logger.info(fold_attr_imp.shape)
        weight_feature_matrix.append(fold_attr_imp_index)
        #if file_count > 2:           
        #    break
        #break

    weight_feature_matrix = np.array(weight_feature_matrix)
    weight_feature_matrix = weight_feature_matrix.astype(int)
    logger.info(weight_feature_matrix.shape)
    start_time = time.time()
    #print weight_feature_matrix
    lda_feature_vector, lda_feature_value_vector = majority_vote_index(weight_feature_matrix, -1)
    #print lda_feature_matrix
    logger.info(lda_feature_vector.shape)
    overall_time = overall_time + time.time() - start_time
    return lda_feature_vector, lda_feature_value_vector, overall_time


def run_cnn_pca_feature_analysis_main(feature_folder, data_folder, data_stru, feature_postfix):
    num_classes = data_stru.num_classes
    start_class = data_stru.start_class
    attr_num = data_stru.attr_num
    class_column = data_stru.class_column

    ret_pca_feature_array = [None] * num_classes
    for i in range(0, num_classes):
        ret_pca_feature_array[i] = []


    file_list = listFiles(data_folder)
    header = True
    overall_time = 0
    file_count = 0
    for train_file in file_list:
        if "train" not in train_file: 
            continue
        print train_file
        #train_file = "train_3.txt"
        file_count = file_count + 1
        train_x_matrix, train_y_vector = readFile(data_folder + train_file)
        [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_folder + train_file+feature_postfix)

        print fold_train_sensor_result.shape
        print fold_weight_fullconn.shape
        print fold_bias_fullconn.shape
        fold_train_sensor_result = np.squeeze(fold_train_sensor_result)
        print fold_train_sensor_result.shape
        
        pca_feature_array = []
        start_time = time.time()
        for i in range(0, num_classes):
            class_label = i + start_class
            print "class: "+ str(class_label)
            class_index = np.where(train_y_vector==class_label)[0]
            class_data_matrix = fold_train_sensor_result[class_index, :, :]
            print class_data_matrix.shape
            print "get error"
            class_im_index, class_im_vector = run_pca_proj_feature_3D(class_data_matrix)
            print "get error" 
            pca_feature_array.append(class_im_index)
            ret_pca_feature_array[i].append(class_im_index)
        pca_feature_array = np.array(pca_feature_array)
        overall_time = overall_time + time.time() - start_time
        
        print pca_feature_array.shape
        #if file_count > 2:
        #    break
        #break

    ret_pca_feature_array = np.array(ret_pca_feature_array)
    print ret_pca_feature_array.shape
    ret_pca_feature_array = ret_pca_feature_array.astype(int)
    start_time = time.time()
    pca_feature_matrix = fold_feature_combination_C_F_A(ret_pca_feature_array)
    overall_time = overall_time + time.time() - start_time
    print pca_feature_matrix.shape
    print pca_feature_matrix[0:5, 0:6]
    return pca_feature_matrix, overall_time


def run_cnn_weight_feature_analysis_main(feature_folder, data_folder, data_stru, feature_postfix):
    num_classes = data_stru.num_classes
    start_class = data_stru.start_class
    attr_num = data_stru.attr_num
    class_column = data_stru.class_column

    ret_weight_feature_array = [None] * num_classes
    for i in range(0, num_classes):
        ret_weight_feature_array[i] = []


    file_list = listFiles(data_folder)
    header = True

    analysis_method = 'projected'

    weight_matrix = np.zeros([num_classes, 10, attr_num])
    overall_time = 0
    fold_count = 0
    for train_file in file_list:
        if "train" not in train_file: 
            continue
        print train_file
        fold_num = train_file[6:7]
        print fold_num

        for i in range(0, num_classes):
            feature_file = feature_folder + train_file + feature_postfix
            print feature_file
            [fold_train_sensor_result, fold_weight_fullconn, fold_bias_fullconn] = load_obj(feature_file)

            #print fold_train_sensor_result.shape
            print "fold_weight_shape"
            print fold_weight_fullconn.shape
            #print fold_bias_fullconn.shape
            start_time = time.time()
            fold_weight_fullconn = np.absolute(fold_weight_fullconn)
            weight_vector = np.sum(fold_weight_fullconn, axis=1)
            print weight_vector
            all_size = len(weight_vector)
            print all_size
            print attr_num
            print "===="
            print len(weight_vector)
            print weight_vector.shape
            print weight_vector
            weight_vector = weight_vector.reshape(attr_num, all_size/attr_num)
            weight_vector = np.sum(weight_vector, axis=1)
            weight_matrix[i, fold_count, :] = top_value_index_from_list(weight_vector, -1)
            overall_time = overall_time + time.time() - start_time

        fold_count = fold_count + 1

        #if fold_count > 2:
        #    break
        #break

    weight_matrix = weight_matrix.astype(int)
    start_time = time.time()
    weight_feature_matrix = fold_feature_combination_C_F_A(weight_matrix)
    overall_time = overall_time + time.time() - start_time
    #print weight_feature_matrix.shape
    #print weight_feature_matrix[0:5, 0:6]
    return weight_feature_matrix, overall_time

if __name__ == '__main__':
    train_x_matrix = np.array([[1, 10000, -1, 1], [1, 9000, -2, 1], [1, 5000, 3, 2], [1, 40000, -1, 1], [1, 30000, 2, 1], [1, 35000, -3, 2]]).astype(float)
    #
    train_y_vector = np.array([1, 1, 1, 2, 2, 3]).astype(int)
    predict = True
    feature_value_vector, lda_model, averaged_acc, run_time = rf_feature_extraction(train_x_matrix, train_y_vector, predict)
    print feature_value_vector, averaged_acc, run_time
    print "======"
    feature_value_vector, lda_model, averaged_acc, run_time = lda_feature_extraction(train_x_matrix, train_y_vector, predict)
    print feature_value_vector, averaged_acc, run_time
    sdfs

    #train_x_matrix = np.random.rand(6, 10, 15)
    #predict = True
#
    ##train_x_matrix = train_x_matrix.reshape(6,2,2)
#
    #project_cnn_feature_combined_rf_lda_analysis(train_x_matrix, train_y_vector)
#
    #sdfsd
    #x_matrix = np.random.rand(100, 5, 4)
    #y_vector = np.zeros(100)
    #y_vector[0:20] = 1
    #logger = init_logging('')
    #lda_weight_feature_matrix = []
    #lda_weight_value_matrix = []
    #map_attr_imp_matrix, skip_count = project_cnn_feature_lda_analysis(x_matrix, y_vector, logger)
    #lda_weight_value_matrix.append(map_attr_imp_matrix)
#
    #x_matrix = np.random.rand(100, 5, 4)
    #y_vector = np.zeros(100)
    #y_vector[0:20] = 1
    #logger = init_logging('')
    #map_attr_imp_matrix, skip_count = project_cnn_feature_lda_analysis(x_matrix, y_vector, logger)
    #lda_weight_value_matrix.append(map_attr_imp_matrix)
#
    #lda_weight_feature_matrix = np.array(lda_weight_feature_matrix)
    #lda_weight_value_matrix = np.array(lda_weight_value_matrix)
#
    #print lda_weight_value_matrix.shape
    #print lda_weight_value_matrix
    #lda_weight_value_matrix = np.sum(lda_weight_value_matrix, axis=1)
    #print lda_weight_value_matrix
    #print np.argsort(-lda_weight_value_matrix, axis=1)
    #asd
    #print "!!!!!"
#
    #print lda_weight_feature_matrix.shape
#
    #print "combination"
    #print lda_weight_feature_matrix
    #ret_feature_matrix = fold_feature_combination_C_F_A(lda_weight_feature_matrix)
    #print ret_feature_matrix

    argv_array = sys.argv
    parameter_file = '../../parameters/project_cnn_feature_generation.txt'

    #print "USAGE: python cnn_project_feature_generation.py <Input_Method> <Parameter_File>"
    #print "USAGE: python cnn_project_feature_generation.py <Input_Method>"

    method = ''
    if len(argv_array)==3:
        method = argv_array[1]
        parameter_file = argv_array[2]
    elif len(argv_array)==2:
        method = argv_array[1]

    data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, new_method, pckl_folder, log_folder, log_postfix, out_obj_folder, feature_postfix = read_project_feature_generation_parameter(parameter_file)
    
    if method == '':
        method = new_method

    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    
    
    log_file = log_folder + data_keyword + '_' + method + log_postfix
    #print 'log file should be: ' + log_file
    #log_file = ''
    #print 'log file actually: ' + log_file
    logger = init_logging(log_file)

    logger.info("data_keyword: " + data_keyword)
    logger.info("data_folder:" + data_folder)
    logger.info("attr_num:" + str(attr_num))
    logger.info("attr_len:" + str(attr_len))
    logger.info("class_column:" + str(class_column))
    logger.info("start_class:" + str(start_class))
    logger.info("num_classes:" + str(num_classes))
    logger.info("method:" + method)
    logger.info("pckl_folder:" + pckl_folder)
    logger.info("log_folder:" + log_folder)
    logger.info("log_postfix:" + log_postfix)
    logger.info("out_obj_folder:" + out_obj_folder)
    logger.info("feature_postfix:" + feature_postfix)

    logger.info('data keyword: '+ data_keyword)
    logger.info('method: '+ method)
    out_obj_file = out_obj_folder + data_keyword + '_' + method + '_no_norm_projected_feature.pckl'
    out_obj_file = out_obj_folder + data_keyword + '_' + method + '_projected_feature.pckl'
    logger.info("out_obj_file: " + out_obj_file)

    
    if method == 'cnn_lda':
        feature_matrix, overall_time = run_cnn_lda_feature_analysis_main(pckl_folder, feature_postfix, data_folder, data_stru, feature_postfix, logger)
        
        save_obj([feature_matrix], out_obj_file)
        logger.info(feature_matrix.shape)
        logger.info(str(feature_matrix[0:5, 0:6]))
        logger.info('output saved at: '+str(out_obj_file))
        logger.info('running time (sec): '+str(overall_time))
    elif method == 'cnn_pca':
        feature_matrix, overall_time = run_cnn_pca_feature_analysis_main(pckl_folder, data_folder, data_stru, feature_postfix)
       
        save_obj([feature_matrix], out_obj_file)
        logger.info(feature_matrix.shape)
        logger.info(str(feature_matrix[0:5, 0:6]))
        logger.info('output saved at: '+str(out_obj_file))
        logger.info('running time (sec): '+str(overall_time))
    elif method == 'cnn_weight':
        feature_matrix, overall_time = run_cnn_weight_feature_analysis_main(pckl_folder, data_folder, data_stru, feature_postfix)
        
        save_obj([feature_matrix], out_obj_file)
        logger.info(feature_matrix.shape)
        logger.info(str(feature_matrix[0:5, 0:6]))
        logger.info('output saved at: '+str(out_obj_file))
        logger.info('running time (sec): '+str(overall_time))
    elif method == 'cnn_rf_class_based':
        feature_matrix = []
        feature_value_matrix = []
        overall_time = 0
        feature_pre = data_keyword + '_'
        for class_label in range(0, num_classes):
            logger.info('Class label: '+ str(class_label))
            feature_array, feature_value_array, class_overall_time = run_cnn_rf_class_based_feature_analysis_main(pckl_folder, feature_postfix, class_label, data_folder, data_stru, logger)
            feature_matrix.append(feature_array)
            feature_value_matrix.append(feature_value_array)
            logger.info('CNN-RF Feature for class ' +str(class_label))
            logger.info(str(feature_array[0:10]))
            logger.info(str(feature_value_array[0:10]))
            overall_time = overall_time + class_overall_time
    elif method == 'cnn_rf_lda_scalings_class_based':
        feature_matrix = []
        feature_value_matrix = []
        overall_time = 0
        feature_pre = data_keyword + '_'
        feature_matrix, feature_value_matrix, overall_time = run_cnn_combined_rf_lda_class_based_feature_analysis_main(pckl_folder, feature_pre, feature_postfix, 0, num_classes, data_folder, data_stru, logger)
        

        #for class_label in range(0, num_classes):
        #    logger.info('Class label: '+ str(class_label))
        #    feature_array, feature_value_array, class_overall_time = run_cnn_combined_rf_lda_class_based_feature_analysis_main(pckl_folder, feature_pre, feature_postfix, 0, class_label, data_folder, data_stru, logger)
        #    feature_matrix.append(feature_array)
        #    feature_value_matrix.append(feature_value_array)
        #    logger.info('CNN-RF-LDA Feature for class ' +str(class_label))
        #    logger.info(str(feature_array[0:10]))
        #    logger.info(str(feature_value_array[0:10]))
        #    overall_time = overall_time + class_overall_time

    elif method == 'cnn_lda_class_based':
        feature_matrix = []
        feature_value_matrix = []
        overall_time = 0
        for class_label in range(0, num_classes):
            logger.info('Class label: '+ str(class_label))
            feature_array, feature_value_array, class_overall_time = run_cnn_lda_class_based_feature_analysis_main(pckl_folder, feature_postfix, class_label, data_folder, data_stru, logger)
            feature_matrix.append(feature_array)
            feature_value_matrix.append(feature_value_array)
            logger.info('CNN-LDA Feature for class ' +str(class_label))
            logger.info(str(feature_array[0:10]))
            logger.info(str(feature_value_array[0:10]))
            overall_time = overall_time + class_overall_time
    elif method == 'cnn_pca_class_based':
        feature_matrix = []
        feature_value_matrix = []
        overall_time = 0
        for class_label in range(0, num_classes):
            logger.info('Class label: '+ str(class_label))
            feature_vector, feature_value_vector, class_overall_time = run_cnn_pca_class_based_feature_analysis_main(pckl_folder, feature_postfix, class_label, data_folder, data_stru, logger)
            feature_matrix.append(feature_vector)
            feature_value_matrix.append(feature_value_vector)
            logger.info('CNN-PCA Feature for class ' +str(class_label))
            logger.info(str(feature_vector[0:10]))
            #logger.info(str(feature_value_vector[0:10]))
            logger.info(str([feature_value_vector[i] for i in feature_vector[0:10]]))
            overall_time = overall_time + class_overall_time


    feature_matrix = np.array(feature_matrix)
    feature_value_matrix = np.array(feature_value_matrix)
    save_obj([feature_matrix, feature_value_matrix], out_obj_file)
    logger.info("Object saved to " + out_obj_file)
    logger.info(feature_matrix.shape)
    logger.info("feature_matrix[0:2, 0:6]")
    logger.info(str(feature_matrix[0:2, 0:6]))
    logger.info("feature_value_matrix[0:2, 0:6]")
    logger.info(str(feature_value_matrix[0:2, 0:6]))
    logger.info('output saved at: '+str(out_obj_file))
    logger.info('running time (sec): '+str(overall_time))

