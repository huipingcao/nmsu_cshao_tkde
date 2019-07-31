import numpy
import sys
import time
import tensorflow as tf
from data_processing import y_vector_to_matrix
import random
from data_io import list_files
from data_io import train_test_file_reading_with_attrnum
from data_io import init_folder
from object_io import save_obj
from object_io import load_obj
from log_io import setup_logger
from sklearn.metrics.pairwise import euclidean_distances
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten


def adam_1(grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    stop = False
    for i in range(num_iters):
        g = grad(x, i)
        if callback: 
            stop = callback(x, i, g)
            if stop == True:
                break
        print type(g)
        #g = np.asarray(g)
        m = (1 - b1) * np.asarray(g)      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (np.asarray(g)**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        print "===Vhat"
        print vhat
        print type(vhat)
        x = x - step_size*mhat/(np.sqrt(np.asarray(vhat)) + eps)
    return x


def mask_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    shapelet = params[0]
    mask = params[1]
    weight = params[2]
    bias = params[3]
    series_matrix = inputs[0]
    series_keep_len = inputs[1]
    dist_a_matrix = min_distance_shapelet_series(series_matrix, shapelet, mask, series_keep_len)
    #print dist_a_matrix.shape
    #print weight.shape
    outputs = np.dot(dist_a_matrix, weight) + bias
    outputs = np.tanh(outputs)
    log_out = logsumexp(outputs, axis=1)
    log_out = log_out.reshape(len(log_out), 1)
    #print log_out.shape
    return outputs - log_out

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(mask_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(mask_predict(params, inputs), axis=1)
    #return np.mean(predicted_class == target_class)
    min_class = min(target_class)
    max_class = max(target_class)
    num_class = max_class - min_class + 1
    acc = 0
    for c in range(min_class, max_class):
        c_index = np.where(target_class==c)[0]
        c_target = target_class[c_index]
        c_pred = predicted_class[c_index]
        acc = acc + np.mean(c_pred == c_target)
    return acc


###
# Used to calculate the A[k, j] in the Arxiv paper
# series: num_series * attr_num * attr_len
# shapelet: num_shap * attr_num * shap_max
# mask: num_shap * attr_num
def min_distance_shapelet_series(series_matrix, shapelet_matrix, mask_matrix, series_keep_len, logger=None):
    if logger is None:
        logger = setup_logger('')
    rs=npr.RandomState(0)
    mask_matrix = np.maximum(0,mask_matrix)
    num_series, attr_num, attr_len = series_matrix.shape
    num_shap, attr_num, shap_max = shapelet_matrix.shape
    ret_a_matrix = []
    for serie_iter in range(0, num_series):
        ret_a_vector = []
        #logger.info("series iter: " + str(serie_iter))
        keep_len = series_keep_len[serie_iter]
        #print series_matrix[serie_iter, :, 0:keep_len]
        series = series_matrix[serie_iter, :, 0:keep_len]
        #logger.info(series.shape)
        for shap_iter in range(0, num_shap):
            #logger.info("shap iter: " + str(shap_iter))
            attr_dist = 0
            for attr in range(attr_num):
                attr_mask = mask_matrix[shap_iter, attr]
                attr_shapelet = shapelet_matrix[shap_iter, attr, :]
                shap_len = len(attr_shapelet)
                attr_series = series[attr, :]
                attr_series_len = len(attr_series)
                attr_shapelet = attr_shapelet.reshape(1, shap_len)
                
                if attr_series_len < shap_len:
                    shap_len = attr_series_len
                    loop_count = 1
                else:
                    loop_count = attr_series_len - shap_len + 1

                #print "loop_count: " + str(loop_count)
                min_euclidean = 0
                start = 0
                end = start + shap_len
                min_euclidean = np.sum(np.square(attr_shapelet - attr_series[start:end]))
                for start in range(1, loop_count):
                    end = start + shap_len
                    euclidean = np.sum(np.square(attr_shapelet - attr_series[start:end]))
                    if euclidean < min_euclidean:
                        min_euclidean = euclidean

                #print "==="

                min_euclidean = min_euclidean / float(shap_len)
                #print min_euclidean
                
                min_euclidean = min_euclidean * attr_mask
                attr_dist = attr_dist + min_euclidean
                #print min_euclidean
                #print attr_mask
                #print attr_dist
            attr_dist = attr_dist / float(attr_num)
            #print serie_iter
            #print shap_iter
            #print "attr dist"
            #print attr_dist
            ret_a_vector.append(attr_dist)
            #ret_a_matrix[serie_iter, shap_iter] = attr_dist[0]
        ret_a_matrix.append(ret_a_vector)
    ret_a_matrix = np.array(ret_a_matrix)
    # print ret_a_matrix.shape
    return ret_a_matrix


###
# train_x_matrix: train_row, attr_num, attr_len
# train_y_matrix: train_row, 2
# shap_k: number of shapelets
# shap_min: min length of shapelets
# shap_max: max length of shapelets
# lamda: regularization
# learning: learning rate
# max_iter: max number of iterations
def run_channel_mask(train_x_matrix, train_y_matrix, train_keep_len, test_x_matrix, test_y_matrix, test_keep_len, shap_k, shap_min, shap_max=-1, logger=None, max_iter=5, learning=0.01, std_value=0.01, L2_reg=0.1):
    
    num_series, attr_num, attr_len = train_x_matrix.shape
    train_row, num_classes = train_y_matrix.shape
    if shap_max == -1:
        shap_max = attr_len
    if num_classes != 2:
        raise Exception("Number of class labels should be 2!")

    rs=npr.RandomState(0)
    shapelet = std_value * rs.randn(shap_k, attr_num, shap_max)
    train_input = [train_x_matrix, train_keep_len]
    test_input = [test_x_matrix, test_keep_len]
    # Training parameters
    param_scale = 0.1
    batch_size = 100
    num_epochs = 5
    step_size = 0.001

    shapelet_matrix_variable = tf.Variable(tf.truncated_normal([shap_k, attr_num, shap_max], stddev=std_value))
    mask_matrix_variable = tf.nn.relu(tf.Variable(tf.truncated_normal([shap_k, attr_num], stddev=std_value)))

    ###
    init_params = []
    # shapelet initilization
    rs=npr.RandomState(0)
    #shapelet = []
    #for i in range(shap_k):
    #    attr_shap = []
    #    for j in range(attr_num):
    #        rand = random.uniform(0, 1)
    #        rand = 0.1
    #        shap_len = int(shap_min + rand * shap_max)
    #        if shap_len > shap_max:
    #            shap_len = shap_max
    #        init_shap = std_value * rs.randn(shap_len)
    #        #print init_shap.shape
    #        attr_shap.append(init_shap)
    #    #print np.array(attr_shap).shape
    #    shapelet.append(attr_shap)
    #shapelet = np.array(shapelet)
    shapelet = std_value * rs.randn(shap_k, attr_num, 12)
    ###
    # other initilization
    mask = std_value * rs.randn(shap_k, attr_num)
    weight = std_value * rs.randn(shap_k, num_classes)
    bias = std_value * rs.randn(num_classes)
    init_params = [shapelet, mask, weight, bias]

    #print init_params[0][0, 1, :]
    #print mask.shape
    #print weight.shape
    #print bias.shape

    num_batches = int(np.ceil(train_row / batch_size))
    logger.info("num batch: " + str(num_batches))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, [train_x_matrix[idx, :, :], train_keep_len[idx]], train_y_matrix[idx, :],  L2_reg)

    #iter_objective = objective(init_params, 1)
    #print "iter objective shape"
    #print iter_objective
    # Get gradient of objective using autograd.

    
    objective_grad = grad(objective)

    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            #train_acc = accuracy(params, train_x_matrix, train_x_label)
            train_acc = 0.0
            test_acc  = accuracy(params, test_input, test_y_matrix)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))
            if test_acc > 0.95:
                return True
        return False

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size, num_iters=num_epochs * num_batches, callback=print_perf)
    test_acc  = accuracy(optimized_params, test_input, test_y_matrix)
    #print test_acc
    #print optimized_params[1].shape
    return test_acc, optimized_params[1]



def run_channel_mask_main(data_folder, log_folder, obj_folder, shap_k=10, shap_min = 2, shap_max=3, file_key="train_", fun_key="_mask_gene"):
    file_list = list_files(data_folder)
    file_count = 0
    for train_file in file_list:
        if file_key not in train_file:
            continue
        this_keyword = train_file.replace('.txt', '')
        log_file = this_keyword + fun_key + "_shapNum" + str(shap_k) + "_shapMin" + str(shap_min) + "_shapMax" + str(shap_max) + "_all_class.log"
        out_obj_file = this_keyword + fun_key + "_shapNum" + str(shap_k) + "_shapMin" + str(shap_min) + "_shapMax" + str(shap_max)
        logger = setup_logger(log_folder + log_file)
        print "log file: " + log_folder + log_file
        print "obj file: " + obj_folder + out_obj_file
        logger.info(log_folder + log_file)
        out_obj_dict = {}
        file_count = file_count + 1
        test_file = train_file.replace('train_', 'test_')
        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(data_folder+train_file, data_folder+test_file)
        
        train_row, train_col = train_x_matrix.shape
        test_row, test_col = test_x_matrix.shape
        attr_len = train_col/attr_num
        train_x_matrix = train_x_matrix.reshape(train_row, attr_num, attr_len)
        test_x_matrix = test_x_matrix.reshape(test_row, attr_num, attr_len)
        logger.info("train x matrix: " + str(train_x_matrix.shape))
        logger.info("test x matrix: " + str(test_x_matrix.shape))

        train_keep_len = matrix_keep_len_gene(train_x_matrix)
        test_keep_len = matrix_keep_len_gene(test_x_matrix)

        min_class = min(train_y_vector)
        max_class = max(train_y_vector) + 1
        num_classes = max_class - min_class
        logger.info("x matrix tran after shape: " + str(train_x_matrix.shape))
        for label in range(min_class, max_class):
            label = max_class - label - 1
            label_train_y_vector = np.where(train_y_vector == label, 1, 0)
            label_test_y_vector = np.where(test_y_vector == label, 1, 0)
            label_train_y_matrix = y_vector_to_matrix(label_train_y_vector, 2)
            label_test_y_matrix = y_vector_to_matrix(label_test_y_vector, 2)
            logger.info("class: " + str(label))
            test_eval_value, mask_value = run_channel_mask(train_x_matrix, label_train_y_matrix, train_keep_len, test_x_matrix, label_test_y_matrix, test_keep_len, shap_k, shap_min, shap_max, logger)
            logger.info("final for class " + str(label))
            logger.info("final acc: " + str(test_eval_value))
            logger.info("final mask: " + str(mask_value.shape))
            logger.info("out obj saved to " + obj_folder + out_obj_file + "_class" + str(label) + ".obj")
            save_obj([mask_value], obj_folder + out_obj_file + "_class" + str(label) + ".obj")

        #train_y_matrix = y_vector_to_matrix(train_y_vector, num_classes)
        #test_y_matrix = y_vector_to_matrix(test_y_vector, num_classes)
        #test_eval_value, mask_value = run_channel_mask(train_x_matrix, train_y_matrix, train_keep_len, test_x_matrix, test_y_matrix, test_keep_len, shap_k, shap_min, shap_max, logger)
        #out_obj_file = out_obj_file + "_allclass.log"
        #logger.info("final for all class ")
        #logger.info("final acc: " + str(test_eval_value))
        #logger.info("final mask: " + str(mask_value.shape))
        #logger.info("out obj saved to " + obj_folder + out_obj_file)
        #save_obj([mask_value], obj_folder + out_obj_file)


def matrix_keep_len_gene(x_matrix):
    x_row, attr_num, attr_len = x_matrix.shape
    ret_vector = []
    for i in range(0, x_row):
        all_keep_len = -1
        for attr in range(0, attr_num):
            x_vector = x_matrix[i, attr, :]
            x_len = len(x_vector)
            keep_len = x_len
            for j in range(0, len(x_vector)):
                re_j = keep_len - j - 1
                if x_vector[re_j] == 0:
                    keep_len = keep_len - 1
                else:
                    break
            if keep_len > all_keep_len:
                all_keep_len = keep_len
        ret_vector.append(all_keep_len)
    return np.array(ret_vector)

def mask_evaluation_main(log_folder, obj_folder, out_obj_folder, obj_keyword, shap_k=-1, shap_min=-1, shap_max=-1, func_key="arxiv_mask_gene"):
    log_folder = log_folder + func_key
    log_folder = init_folder(log_folder)
    log_file = obj_keyword + "_allclass_" + func_key + ".log"
    #logger = setup_logger('')
    logger = setup_logger(log_folder + log_file)
    logger.info("log folder: " + log_folder)
    logger.info("obj folder: " + obj_folder)
    obj_file_list = list_files(obj_folder)
    
    if shap_k != -1:
        obj_sec_key = "shapNum" + str(shap_k) + "_shapMin" + str(shap_min) + "_shapMax" + str(shap_max)
    else:
        obj_sec_key = ".obj"
    min_class = 100
    max_class = -1
    output_array = []
    
    for obj_file in obj_file_list:
        if obj_keyword not in obj_file:
            continue
        if "_class" not in obj_file:
            continue
        if obj_sec_key not in obj_file:
            continue
        class_key = obj_file.split('_')[-1]
        class_key = class_key.replace('class', '').replace('.obj', '')
        logger.info("obj file:" + obj_file)
        logger.info("class key: " + class_key)
        class_key = int(class_key)
        if min_class > class_key:
            min_class = class_key
        if max_class < class_key:
            max_class = class_key
        shap_mask = load_obj(obj_folder + obj_file)[0]
        if len(shap_mask) == 0:
            continue
        shap_mask = numpy.array(shap_mask)
        shap_mask = numpy.squeeze(shap_mask)
        logger.info("shap_mask shape: " + str(shap_mask.shape))
        #shap_num, attr_num = shap_mask.shape
        
        shap_mask = numpy.absolute(shap_mask)
        shap_mask = numpy.sum(shap_mask, axis=0)
        logger.info(shap_mask)
        sort_index = numpy.argsort(shap_mask)
        imp_value = 0
        norm_imp = numpy.zeros(len(shap_mask))
        for index in sort_index:
            norm_imp[index] = imp_value
            imp_value = imp_value + 1
        shap_mask_index = numpy.argsort(norm_imp)[::-1]
        logger.info(shap_mask_index)
        logger.info("====")
        output_array.append(shap_mask_index)
        logger.info("shap_mask final shape: " + str(shap_mask.shape))
    output_array = numpy.array(output_array)
    obj_file = obj_keyword + "_min" + str(min_class) + "_max" + str(max_class) + "out.obj"
    logger.info("final output obj shape: " + str(output_array.shape))
    logger.info(output_array)
    save_obj([output_array], out_obj_folder + obj_file)

        



if __name__ == '__main__':
    #for i in range(1000):
    #    a = np.random.rand(1, 12)
    #    b = np.random.rand(46, 12)
    #    print a.shape
    #    print b.shape
    #    c = euclidean_distances(a, b)[0]
    #    print c.shape
    #    print a.shape
    #sfss


    data_keyword = 'dsa'
    #data_keyword = 'rar'
    #data_keyword = 'arc'
    data_keyword = 'ara'
    data_keyword = 'asl'
    keyword = "arxiv_2017_mask"
    data_sub_folder = "train_test_1_fold"
    data_sub_folder = "train_test_3_fold"
    data_folder = '../../data/' + data_keyword + '/' + data_sub_folder + '/'
    obj_folder = '../../object/' + data_keyword + '/' + keyword +'/' 
    obj_folder = init_folder(obj_folder)
    log_folder = '../../log/' + data_keyword + '/' + keyword + '/'
    log_folder = init_folder(log_folder)
    file_key = 'train_'
    argv_array = sys.argv
    len_argv_array = len(argv_array)
    if len_argv_array > 1:
        try:
            val = int(argv_array[1])
            file_key = file_key + argv_array[1]
        except ValueError:
            print("That's not an int!")
    method_key = 0 # Generate the shap-mask object from original datasets
    method_key = 1 # Generate the attr-importance from the generated shap-mask objects

    shap_k = 10
    shap_min = 3
    shap_max = 5
    if method_key == 0:
        run_channel_mask_main(data_folder, log_folder, obj_folder, shap_k, shap_min, shap_max, file_key)
    elif method_key == 1:
        func_key="arxiv_mask_gene"
        out_obj_folder = obj_folder.replace(keyword, func_key)
        out_obj_folder = init_folder(out_obj_folder)
        mask_evaluation_main(log_folder, obj_folder, out_obj_folder, file_key, shap_k, shap_min, shap_max, func_key)
