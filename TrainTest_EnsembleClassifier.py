import os
import tensorflow as tf
import numpy as np
import Confs as Confs
from TrainTest_Interface import train_models, train_ensemble, test, train_base
from Utils import weighted_loss_tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# IMPORT ENDAVA DATASET SETTINGS FROM Confs.py
dataset_folder = Confs.parameters_general['dataset_folder']
dataset_name = Confs.parameters_general['dataset_name']
text_col_name = Confs.parameters_general['text_col_name']
label_col_name = Confs.parameters_general['label_col_name']
num_classes = Confs.parameters_general['num_classes']

def single_test(seed, results_path):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    Confs.parameters_general['num_classes'] = num_classes
    if num_classes == 2:
        #Confs.parameters_general['loss'] = 'binary_crossentropy'
        Confs.parameters_general['loss'] = weighted_loss_tf
        #Confs.parameters_ensemble["loss"] = 'binary_crossentropy'
        Confs.parameters_ensemble["loss"] = weighted_loss_tf
        Confs.parameters_general['dim_last_dense'] = 4  # 1 for binary classification - 4 for 4 classes
        Confs.parameters_general['last_activation'] = 'sigmoid'  # 'sigmoid' for binary classification - 'softmax' for 4 classes
    elif num_classes > 2:
        Confs.parameters_general['loss'] = 'categorical_crossentropy'
        Confs.parameters_ensemble["loss"] = 'categorical_crossentropy'
        Confs.parameters_general['dim_last_dense'] = num_classes  # 1 for binary classification - 4 for 4 classes
        Confs.parameters_general['last_activation'] = 'softmax'  # 'sigmoid' for binary classification - 'softmax' for 4 classes

    base_learner_names = ['transformer', 'gru', 'lstm', 'cnn']
    combiners_mode = ['stacking_ffnn', 'moe']

    aggressive_opt = Confs.parameters_general['aggressive_opt']
    train_models(dataset_name, results_path, base_learner_names,
                    aggressive_opt=aggressive_opt, text_col_name=text_col_name,
                    label_col_name=label_col_name)
    train_base(dataset_name, results_path, base_models_path=None, text_col_name=text_col_name,
               label_col_name=label_col_name)
    for combiner_mode in combiners_mode:
        print("Training the ensemble model " + "ensemble_" + combiner_mode)
        train_ensemble(dataset_name, results_path, combiner_mode, aggressive_opt=aggressive_opt,
                           text_col_name=text_col_name, label_col_name=label_col_name)
        # name of predictor to use to predict labels on test set
        predictor_name = "ensemble_model/ensemble_" + combiner_mode + ".h5"
        # path to the chosen prediction model
        predictor_path = os.path.join(results_path, predictor_name)
        print("Using the " + predictor_name + " to re-label the test set...")
        test(dataset_name, results_path, predictor_path, text_col_name=text_col_name)
        print("The re-labeled test set has been generated correctly!")

if __name__ == "__main__":
    num_test = 20
    seed = Confs.parameters_general["seed"]
    np.random.seed(seed)
    seeds =  np.random.rand(num_test)
    f_all_results = open(Confs.parameters_general['result_path_file'], "w")
    f_all_results.write("model\tacc\tauc\tg-mean\tF1\n")
    f_all_results.close()
    for i in range (0, num_test):
        print('****************** TEST ' + str(i) + ' ******************')
        results_path = './tests/' + dataset_folder + '/results_' + str(i) + '/'
        current_seed = int(seeds[i]*100)
        Confs.parameters_general['strat_shuffle_seed'] = current_seed
        print('strat_shuffle_seed' + ':' + str(current_seed))
        single_test(current_seed, results_path)
