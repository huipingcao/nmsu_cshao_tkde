import numpy as np
from log_io import init_logging
from sklearn.decomposition import PCA
from numpy import mean, std
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin_min
from data_io import file_read_split
from data_io import init_folder
from data_io import list_files
from object_io import save_obj
from object_io import load_obj
import operator
###

# Do standardization
def standardization(data_matrix):
    data_matrix -= mean(data_matrix, 0)
    data_matrix /= std(data_matrix, 0)
    return data_matrix

# PCA analysis using sklearn pca package
# data_matrix: 2D matrix, N * C
# K: Top K PCA
# Return: N * K
def sklearn_pca_analysis(data_matrix, K):
    pca = PCA(n_components=K)
    pca.fit(data_matrix)
    return pca.transform(data_matrix), pca.components_.T


def computeDCPC(mts_data, threshold=0.9, logger=None):
    if logger==None:
        logger = init_logging('')
    
    row_num, attr_len, attr_num = mts_data.shape
    print mts_data.shape
    loading = []
    percent = []
    for r in range(0, row_num):
        mts_item = mts_data[r, :, :]
        #mts_item = standardization(mts_item)
        #logger.info("mts item: " + str(mts_item.shape))
        corr_matrix = np.corrcoef(mts_item)
        indices = np.where(np.isnan(corr_matrix))
        corr_matrix[indices]=0
        #logger.info("corr_matrix: " + str(corr_matrix.shape))
        u, s, vh = np.linalg.svd(corr_matrix, full_matrices=True)
        percent_var = (s/sum(s))
        p_sum = float(0)
        for p in range(0, len(percent_var)):
            p_sum = p_sum + percent_var[p]
            if p_sum >= threshold:
                break
        percent.append(p)
        #logger.info("u: " + str(u.shape))
        loading.append(u)
    p = max(percent)

    h_matrix = []
    for r in range(0, row_num):
        load_m = loading[r]
        mul_load = np.multiply(load_m.T, load_m)
        if len(h_matrix) == 0:
            h_matrix = mul_load
        else:
            h_matrix = h_matrix + mul_load
    
    logger.info(h_matrix.shape)
    indices = np.where(np.isnan(h_matrix))
    h_matrix[indices]=0
    dcpc, h_s, h_v = np.linalg.svd(h_matrix, full_matrices=True)
    logger.info(dcpc[0:p, :])
    return dcpc[0:p, :]


def clever_rank(dcpc, logger=None):
    if logger is None:
        logger = init_logging('')
    top_p, attr_num = dcpc.shape
    attr_score = {}
    for a in range(0, attr_num):
        var_score = dcpc[:, a]
        l2_score = np.sum(np.square(var_score))
        attr_score[a] = l2_score
    return attr_score


def clever_cluster(dcpc, k, logger=None):
    if logger == None:
        logger = init_logging('')
    
    keep_model = None
    keep_dis = -1
    dcpc = dcpc.T
    for i in range(0, 20):
        model = KMeans(n_clusters=k).fit(dcpc)
        centers = np.array(model.cluster_centers_)
        labels = model.labels_
        overall_dis = 0
        for label in range(0, k):
            clu_idx = np.where(labels==label)[0]
            if len(clu_idx) == 0:
                continue
            clu_ins = []
            for idx in clu_idx:
                clu_ins.append(dcpc[idx, :])
            clu_ins = np.array(clu_ins)
            center_label = centers[label]
            center_label = center_label.reshape(1, len(center_label))
            clu_dis = euclidean_distances(clu_ins, center_label)
            clu_dis = np.sum(clu_dis)
            overall_dis = overall_dis + clu_dis
        if keep_dis < 0 or keep_dis > overall_dis:
            keep_model = model
            keep_dis = overall_dis
            print model.labels_
    
    closest, _ = pairwise_distances_argmin_min(keep_model.cluster_centers_, dcpc)
    
    return closest

def run_dcpc_main(data_folder, class_column, num_classes, obj_folder, threshold, logger=None):
    if logger == None:
        logger = init_logging('')

    file_list = list_files(data_folder)
    overall_time = 0

    file_count = 0
    out_obj_dict = {}
    for train_file in file_list:
        if "train_" not in train_file: 
            continue
        logger.info(train_file)
        out_obj_file = train_file.replace('.txt', '_dcpc.obj')
        file_count = file_count + 1
        
        test_file = train_file.replace('train_', 'test_')

        x_matrix, y_vector = file_read_split(data_folder + train_file)
        min_class = min(y_vector)
        max_class = max(y_vector) +1
        #logger.info("x matrix tran after shape: " + str(x_matrix.shape))
        #x_matrix = x_matrix.transpose((0, 2, 1))
        logger.info("x matrix tran after shape: " + str(x_matrix.shape))
        for label in range(min_class, max_class):
            label_index = np.where(y_vector==label)[0]
            label_x_matrix = x_matrix[label_index, :, :]
            logger.info("class: " + str(label))
            print "class: " + str(label)
            logger.info("x matrix tran before shape: " + str(label_x_matrix.shape))
            label_dcpc = computeDCPC(label_x_matrix, threshold)
            logger.info("class: " + str(label) + " dcpc shape: " + str(label_dcpc.shape))
            out_obj_dict[label] = label_dcpc
        logger.info("dcpc out obj: " + str(obj_folder + out_obj_file))
        save_obj([out_obj_dict], obj_folder + out_obj_file)



def run_dcpc_processing(dcpc_folder, num_classes, method=0, logger=None):
    logger.info('obj folder:' + dcpc_folder)
    dcpc_list = list_files(dcpc_folder)
    logger.info(dcpc_list)
    score_folder = dcpc_folder[:-1] + "_score/"
    score_folder = init_folder(score_folder)
    for dcpc_obj in dcpc_list:
        dcpc = load_obj(dcpc_folder + dcpc_obj)[0]
        if method == 0:
            out_label_array = []
            out_label_dict = {}
            for label in range(0, num_classes):
                logger.info('class: ' + str(label))
                label_dcpc = dcpc[label]
                logger.info("dcpc shape: " + str(label_dcpc.shape))
                attr_score = clever_rank(label_dcpc, logger)
                logger.info(attr_score)
                sorted_dict = sorted(attr_score.items(), key=operator.itemgetter(1), reverse=True)
                sorted_attr = []
                for item in sorted_dict:
                    sorted_attr.append(item[0])
                #label_array = []
                #for label in range(0, num_classes):
                #    class_array = sorted_attr
                #    label_array.append(class_array)
                out_label_array.append(sorted_attr)
                out_label_dict[label] = attr_score
                logger.info(sorted_attr)
                logger.info(attr_score)
            save_obj([out_label_array, out_label_dict], score_folder + dcpc_obj)
            
            logger.info("score obj: " + score_folder + dcpc_obj)

    return score_folder


if __name__ == '__main__':

    #train_x_matrix = np.random.rand(10, 4, 20)
    #threshold = 0.99
    #k = 2
    #clever_rank(train_x_matrix, k, threshold)
    #dcpc = computeDCPC(train_x_matrix, threshold)
    #clever_cluster(dcpc, k)
    #sdfs
    
    data_keyword = 'dsa'
    data_keyword = 'rar'
    data_keyword = 'arc'
    data_keyword = 'asl'
    data_keyword = 'fixed_arc'

    data_folder = '../../data/' + data_keyword + '/train_test_10_fold/'
    data_folder = '../../data/' + data_keyword + '/train_test_3_fold/'
    data_folder = '../../data/' + data_keyword + '/train_test_1_fold/'
    class_column = 0
    num_classes = 18
    threshold = 0.9
    obj_folder = '../../object/' + data_keyword + '/tkde_2005_dcpc/'
    obj_folder = init_folder(obj_folder)
    log_folder = '../../log/' + data_keyword + '/tkde_2005/'
    log_folder = init_folder(log_folder)
    #log_file = log_folder + data_keyword + "_tkde_dcpc.log"
    #logger = init_logging(log_file)
    #run_dcpc_main(data_folder, class_column, num_classes, obj_folder, threshold, logger)

    method = 0
    log_file = log_folder + data_keyword + "_dcpc_to_score.log"
    logger = init_logging(log_file)
    run_dcpc_processing(obj_folder, num_classes,  method, logger)



