import os
import tensorflow as tf
import Confs as Confs
from explanationUtils import create_testset_trainset, \
    create_lat_rep, find_k_neighbours, limeExplain, wordCloudsExplain

# IMPORT ENDAVA DATASET SETTINGS FROM Confs.py
dataset_folder = Confs.parameters_general['dataset_folder']
text_col_name = Confs.parameters_general['text_col_name']
label_col_name = Confs.parameters_general['label_col_name']
dataset_name = Confs.parameters_general['dataset_name']


if __name__ == "__main__":
    results_path = './tests/' + dataset_folder + '/results_' + str(0) + '/'
    # model to be used for explanation
    ensemble_model = 'stacking_ffnn'
    predictor_name = "ensemble_model/ensemble_" + ensemble_model + ".h5"
    # path to the chosen prediction model
    predictor_path = os.path.join(results_path, predictor_name)

    # OUTPUT FILES
    trainset_file = "trainset.csv"
    testset_file = "testset.csv"

    latent_representation_train_file = ensemble_model + "_preds_lat_rep_train.csv"
    latent_representation_test_file = ensemble_model + "_preds_lat_rep_test.csv"

    knn_file_name = 'k_nearest_neighbors.csv'

    # k = number of neighbours
    k = 100
    # n_words = number of words to be processed for word_cloud explanation
    n_words=100
    # target_ticket = number of ticket in the testset to be explained
    target_ticket = 100
    '''
    # CREATE trainset.csv and testset.csv
    create_testset_trainset(dataset_name, results_path, text_col_name, label_col_name, trainset_file, testset_file)
    '''

    '''
    #CREATE LATENT REPRESENTATION FOR trainset.csv and testset.csv
    # output files: ensemble_stacking_ffnn_preds_lat_rep_test.csv , ensemble_stacking_ffnn_preds_lat_rep_train.csv
    file_name = trainset_file
    file_path = os.path.join(results_path, file_name)
    print(file_path)
    create_lat_rep(file_path, results_path, predictor_path, text_col_name, latent_representation_train_file, label_col_name, lat_rep_col_name='predicted_lat_rep')

    file_name = testset_file
    file_path = os.path.join(results_path, file_name)
    print(file_path)
    create_lat_rep(file_path, results_path, predictor_path, text_col_name, latent_representation_test_file,label_col_name, lat_rep_col_name='predicted_lat_rep')
    '''


    # FIND NEIGHBOURS IN THE LATENT REPRESENTATION SPACE - OUTPUT FILE = k_nearest_neighbors.csv
    find_k_neighbours(latent_representation_train_file, latent_representation_test_file, results_path, target_ticket, k, text_col_name,
                          label_col_name, knn_file_name, lat_rep_col_name='predicted_lat_rep')

    # EXPLAIN TARGET TICKET WITH WORD CLOUD
    wordCloudsExplain(knn_file_name, label_col_name, text_col_name, results_path, n_words)

    # EXPLAIN TARGET TICKET WITH LIME
    limeExplain(testset_file,target_ticket, predictor_name, text_col_name, label_col_name, results_path)
    exit(0)





