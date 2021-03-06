# nmsu_cshao_tkde
The code for Yifan Hao's TKDE paper
This is the code repository for paper: Yifan Hao, Huiping Cao, Abdullah Mueen, Sukumar Brahma: Identify Significant Phenomenon-specific Variables for Multivariate Time Series

Data link: ask for download link. Yifan Hao: yifan@nmsu.edu

Running Example using the toy dataset:
For example, the dataset name is "toy" and there are two data files under data/toy folder: train_0.txt and test_0.txt
1. Run PV generation using CNN_${mts}$

    1.1 Scripts:
    # python pv_cnn_generation.py 0

    1.2 Outputs:  
    The output locates on object/toy/pv_cnn_generation/  

2. Run PV evaluation based on the output objects from step 1  
    2.1 Script:  
    # python pv_cnn_evaluation.py toy rf_lda 0  
    2.2 Parameters:  
    "toy": is the data folder name  
    "rf_lad" is the evaluation method name  
    "0" is an optional parameter. It identify which fold to run. The program will run all folds if the parameter is missing.  
    2.3 Outputs:  
    The output object file contains the orderd PVs  
    2.4 Others:  
    For other cnn based baselines, those can be runned using different method parameters. For example, use "rf" instead of "rf_lda"  
    
3. The PVs can be used in either binary-classifications or multi-class classifications.  
    3.1 Scripts:  
    # python pv_classification.py 0  
    3.2 Parameters:  
    "0" is an optional parameter. It identify which fold to run. The program will run all folds if the parameter is missing.  

4. For the PV generation without CNN_${mts}$  
    4.1 Script:  
    # python pv_baseline_evaluation.y 0  
    4.2 Parameters:  
    "0" is an optional parameter. It identify which fold to run. The program will run all folds if the parameter is missing.  

5. For the global variables generation based on PV
    5.1 Script:  
    # python global_feature_generation.py  
    5.2 Parameters: The parameter file is global_feature_generation.txt

6. For the multi-class classification using PVs  
    6.1 Script:  
    # python multi_proj_feature_classification.y 0  
    6.2 Parameters:  
    "0" is an optional parameter. It identify which fold to run. The program will run all folds if the parameter is missing.  
    This program uses the same PVs identified above  
    Other parameters are from the parameter file: pv_classification.txt  

7. Other baselines  
    7.1 Forward wrapper  
    # python forward_multitime.py 0  
    7.2 Backward wrapper  
    # python backward_multitime.py 0  
    7.3 Best wrapper  
    # python best_forward_multitime.py 0  
    7.4 Channel mask  
    # python arxiv_2017_channel_mask.py 0  
    7.5 cpca
    # python tkde_2005_pca.py 0
