import pickle
from collections import Counter
import numpy as np
from data_io import list_files

def save_obj(obj_list, out_file):
    with open(out_file, 'w') as f:
        pickle.dump(obj_list, f)
        

def load_obj(load_file):
    with open(load_file) as f:
        obj_list = pickle.load(f)
    return obj_list

if __name__ == '__main__':

    obj_file = "../../object/dsa/incre_genetic_projected_cnn_out_pckl/train_0_incre_genetic_projected_fit0_class0_result.pckl"
    obj_file = "../../object/dsa/incre_genetic_projected_cnn_out_pckl/train_5_incre_genetic_projected_fit0_class1_result.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/incre_genetic_projected_cnn_out_pckl/train_2_incre_genetic_projected_fit0_class0_result.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/projected_generation/genetic_cnn_projected_class_based_class6.pckl"
    obj_file = "../../object/dsa/incre_genetic_projected_cnn_temp_saver/train_5.txt_class8_fit_type_0_gene_3__c5_1_p2_1_c-1_1_p-1_-1.ckpt_last_conv_layer_output.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/projected_load_generation/genetic_cnn_projected_class_based_class6.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/incre_genetic_projected_9_cnn_out_pckl/train_0_incre_genetic_projected_9_fit9_class6_result.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/incre_genetic_projected_9_cnn_out_pckl_all_fold_knn/incre_genetic_projected_9_fit9_class1_result.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/incre_genetic_projected_9_cnn_out_pckl/train_1_incre_genetic_projected_9_fit9_class6_result.pckl"
    obj_file = "../../object/dsa/train_test_3_fold/incre_genetic_covered_fit9_fold_feature/train_2_incre_genetic_covered_fit9_fit9_class9_result.pckl"
    obj_file = '../../object/dsa/tkde_2005_dcpc_score/train_0_dcpc.obj'
    obj_file = "../../object/rar/all_feature_classification/fcn_obj_folder_rf_lda/train_6_rf_lda_min0_max33.obj"
    obj_file = "../../object/arabic/arxiv_2017_mask/train_0_mask_gene_shapNum8_shapMin10_shapMax20_class0.obj"
    obj_file = "../../object/dsa/all_feature_classification/cnn_result_folder/train_0_all_feature_cnn_result.ckpt"
    obj_file = "../../object/dsa/all_feature_classification/cnn_obj_folder/train_9_count0_cnn_class0_c5_1_p2_1_c5_1_p2_1_c115_1_p-1_-1.ckpt"
    obj_file = "../../object/dsa/all_feature_classification/fcn_obj_folder/train_9_count0_fcn_class0_c8_1_c5_1_c3_1global_p112_1.ckpt"
    obj_file = "../../object/ara/arxiv_2017_mask/train_0_mask_gene_shapNum8_shapMin10_shapMax20_class1.obj"
    obj_file = "../../object/ara/arxiv_2017_mask/train_0_mask_gene_shapNum10_shapMin3_shapMax5_class0.obj"
    obj_file = "../../object/ara/pure_feature_generation/train_0_rf_lda_min0_max10pure_projected.obj"
    obj_file = "../../object/dsa/forward_wrapper/dsa_train_0_forward_wrapper_class-1_rf.out"
    obj_file = "/home/ivan/Research/projects/nmsu_cshao_tkde/object/toy/pv_cnn_generation/cnn_obj_folder_rf_lda/train_0_rf_lda_min0_max19.obj"
    #obj_folder = '../../object/dsa/tkde_2005_dcpc_score_libsvm_out/'
    #obj_folder = '../../object/dsa/all_feature_classification/fcn_obj_folder/'
    #obj_file = 'train_0_count0_fcn_class0_c8_1_c5_1_c3_1global_p112_1.ckpt'
    #obj_vector = load_obj(obj_folder + obj_file)
    obj_vector = load_obj(obj_file)[0]
    print obj_vector.shape
    print obj_vector
    sdfs
    print np.array(obj_vector[0]).shape
    print np.array(obj_vector[0]).shape
    print len(obj_vector)
    print np.array(obj_vector[0]).shape
    #print np.array(obj_vector[0][1]).shape
    #print np.array(obj_vector[0][8]).shape
    #print np.array(obj_vector[1]).shape
    sdfds
    obj_list = list_files(obj_folder)
    acc_vector = []
    train_vector = []
    test_vector = []
    obj_count = 0
    for obj_file in obj_list:
        print obj_file
        obj_vector = load_obj(obj_folder + obj_file)
        #print obj_vector[0]
        #print obj_vector[3]
        acc_vector.append(float(obj_vector[0]))
        train_vector.append((obj_vector[3]))
        test_vector.append((obj_vector[4]))
        obj_count = obj_count + 1
    
    acc_vector = np.array(acc_vector)
    train_vector = np.array(train_vector)
    test_vector = np.array(test_vector)

    #print acc_vector
    #print train_vector
    #print test_vector
    print acc_vector.shape
    print sum(acc_vector)/obj_count



    #print obj_list[0].selected_list
    #print obj_list[0].eval_value
    #print obj_list[1].selected_list
    #print obj_list[1].eval_value
    #print obj_list[2].selected_list
    #print obj_list[2].eval_value
    
    
