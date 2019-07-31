import numpy as np
from sklearn.metrics import accuracy_score
from log_io import init_logging


def averaged_class_based_accuracy(predict_y_vector, real_y_vector):
    min_class = min(real_y_vector)
    max_class = max(real_y_vector) + 1

    averaged_accuracy = 0
    ret_str = "class based accuracy: "
    for c in range(min_class, max_class):
        class_index = np.where(real_y_vector==c)
        class_predict_y = predict_y_vector[class_index]
        class_real_y = real_y_vector[class_index]
        class_accuracy = accuracy_score(class_real_y, class_predict_y, True)
        ret_str = ret_str + str(c) + ":" + str(class_accuracy) + " "
        averaged_accuracy = averaged_accuracy + class_accuracy
        
    return float(averaged_accuracy)/(max_class-min_class), ret_str



def predict_matrix_with_prob_to_predict_accuracy(predict_prob_matrix, real_y_vector):
    predict_y_vector = np.argmax(predict_prob_matrix, axis=1)
    return accuracy_score(real_y_vector, predict_y_vector), predict_y_vector

# For binary class classification only
# 1 means positive class and 0 means negative class label
def f1_value_precision_recall_accuracy(predict_y_vector, real_y_vector, major_class=1):
    if len(predict_y_vector) != len(real_y_vector):
        raise Exception("Length for prediction is not same")
    #if (max(predict_y_vector) != max(real_y_vector)) or (min(predict_y_vector) != min(real_y_vector)):
    #    raise Exception("max or min prediction is not same")

    instance_num = len(predict_y_vector)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    accuracy = 0
    for i in range(0, instance_num):
        predict = int(predict_y_vector[i])
        real = int (real_y_vector[i])
        if real == major_class:
            if predict == real:
                tp = tp + 1
                accuracy = accuracy + 1
            else:
                fn = fn + 1
        else:
            if predict == real:
                tn = tn + 1
                accuracy = accuracy + 1
            else:
                fp = fp + 1

    accuracy = float(accuracy) / float(instance_num)
    if tp == 0:
        precision = 0
        recall = 0
        f1_value = 0
    else:
        precision = float(tp)/float(tp + fp)

        recall = float(tp)/float(tp + fn)

        f1_value = float(2 * precision * recall) / float(precision + recall)

    return accuracy, precision, recall, f1_value, tp, fp, tn, fn


def multiple_f1_value_precision_recall_accuracy(predict_y_vector, real_y_vector, logger=None):
    if logger == None:
        logger = init_logging('')
    if len(predict_y_vector) != len(real_y_vector):
        raise Exception("Length for prediction is not same")

    min_class = min(real_y_vector)
    max_class = max(real_y_vector)

    instance_num = len(predict_y_vector)
    f1_value_list = []
    for i in range(min_class, max_class+1):
        class_predict_y = np.where(predict_y_vector == i, 1, 0)
        class_real_y = np.where(real_y_vector ==i, 1, 0)
        #print class_predict_y
        #print class_real_y
        #print "==="
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for instance_index in range(0, instance_num):
            predict = int(class_predict_y[instance_index])
            real = int (class_real_y[instance_index])
            if real == 1:
                if predict == real:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if predict == real:
                    tn = tn + 1
                else:
                    fp = fp + 1
    
        if tp == 0:
            precision = 0
            recall = 0
            f1_value = 0
        else:
            precision = float(tp)/float(tp + fp)
            recall = float(tp)/float(tp + fn)
            f1_value = float(2 * precision * recall) / float(precision + recall)
        f1_value_list.append(f1_value)
    f1_value_list = np.array(f1_value_list)

    accuracy = 0
    for instance_index in range(0, instance_num):
        predict = int(predict_y_vector[instance_index])
        real = int (real_y_vector[instance_index])
        if predict == real:
            accuracy = accuracy + 1
    accuracy = float(accuracy)/float(instance_num)
    return accuracy, f1_value_list

if __name__ == '__main__':
    real_y_vector = np.array([0,0,0,1,1,1,1,1,1,1])
    predict_y_vector = np.array([1,0,0,2,1,1,0,2,2,3])
    #predict_y_vector = np.array([0,0,0,0,0,0,0,0,0,0])
    #predict_y_vector = np.array([1,1,1,1,1,1,1,1,1,1])
    averaged_class_based_accuracy(predict_y_vector, real_y_vector)

    sdf
    accuracy, f1_value_list = multiple_f1_value_precision_recall_accuracy(predict_y_vector, real_y_vector)
    print accuracy
    print f1_value_list
