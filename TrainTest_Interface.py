import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix, auc
import numpy as np
from imblearn.metrics import geometric_mean_score

import Confs as Confs
from BuildBaseModels import build_base_models
from BuildEnsembleNN import build_ensemble_model
from BuildInputTensors import extract_test_tensors, strat_shuffle_split_training, extract_tensor
from BuildInputTensors import extract_train_tensors, read_training_data_tensors, split_training
from Utils import compute_accuracy, compute_AUC
from Utils import load_model_and_compile, load_models_and_compile


def train_models(dataset_file, results_path, model_names, text_col_name,  label_col_name, aggressive_opt=True):
    csv_separator_char = Confs.parameters_general["csv_separator_char_training"]

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    datatensors_path = os.path.join(results_path, 'input_tensors/')

    # Read input data
    if os.path.exists(os.path.join(datatensors_path, "data.npy")):
        data, labels = read_training_data_tensors(datatensors_path)
    else:
        extract_train_tensors(dataset_file, datatensors_path, csv_separator_char, text_col_name=text_col_name,
                              label_col_name=label_col_name)
        data, labels = read_training_data_tensors(datatensors_path)

    # Split input data
    x_train_base, y_train_base, _, _, x_test, y_test = strat_shuffle_split_training(data, labels)

    # Build base models
    valPerc = Confs.parameters_general["valPerc"]

    print("Building base models...")
    base_models = build_base_models(x_train_base, y_train_base, x_test, y_test, results_path, model_names,
                                    aggressive_opt, valPerc=valPerc)

def eval_single_model(y_test, y_pred, cut_off=0.5, decimal_numbers=5):
    num_classes = Confs.parameters_general['num_classes']
    if num_classes == 2:
        y_pred = np.ravel(y_pred)
        y_pred_round = y_pred >= cut_off
        acc = round(accuracy_score(y_test, y_pred_round), decimal_numbers)
        auc_score = round(roc_auc_score(y_test, y_pred), decimal_numbers)
        g_mean = round(geometric_mean_score(y_test, y_pred_round), decimal_numbers)
        f1 = round(f1_score(y_test, y_pred_round), decimal_numbers)
        pr1, rec1, thr1 = precision_recall_curve(y_test, y_pred)
        auc_score_pr = auc(rec1, pr1)
        print(confusion_matrix(y_test, y_pred_round))
        conf_matrix_file = Confs.parameters_general['conf_matrix_file']
        f_conf_matrix_file = open(conf_matrix_file, "a+")
        f_conf_matrix_file.write(str(confusion_matrix(y_test, y_pred_round)))
        f_conf_matrix_file.write('\n')
        f_conf_matrix_file.close()
    else:
        y_pred_vec = np.argmax(y_pred, axis=1)
        y_test_vec = np.argmax(y_test, axis=1)
        acc = round(accuracy_score(y_test_vec, y_pred_vec), decimal_numbers)
        auc_score = round(roc_auc_score(y_test, y_pred, multi_class='ovo'), decimal_numbers)
        g_mean = round(geometric_mean_score(y_test_vec, y_pred_vec, average='macro'), decimal_numbers)
        f1 = round(f1_score(y_test_vec, y_pred_vec, average='macro'), decimal_numbers)
        print(confusion_matrix(y_test_vec, y_pred_vec))
        conf_matrix_file = Confs.parameters_general['conf_matrix_file']
        f_conf_matrix_file = open(conf_matrix_file, "a+")
        f_conf_matrix_file.write(str(confusion_matrix(y_test_vec, y_pred_vec)))
        f_conf_matrix_file.write('\n')
        f_conf_matrix_file.close()
    return ''.join(["{:0.5f}".format(acc), "\t", "{:0.5f}".format(auc_score), "\t", "{:0.5f}".format(g_mean), "\t", "{:0.5f}".format(f1)])

def train_ensemble(dataset_file, results_path, combiner_mode, base_models_path=None, aggressive_opt=True,
                   text_col_name='SentimentText', label_col_name='label_col_name'):
    csv_separator_char = Confs.parameters_general["csv_separator_char_training"]

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if base_models_path is None:
        base_models_path = os.path.join(results_path, "base_models/")

    datatensors_path = os.path.join(results_path, 'input_tensors/')

    # Read input data
    if os.path.exists(os.path.join(datatensors_path, "data.npy")):
        data, labels = read_training_data_tensors(datatensors_path)
    else:
        extract_train_tensors(dataset_file, datatensors_path, csv_separator_char, text_col_name=text_col_name,
                              label_col_name=label_col_name)
        data, labels = read_training_data_tensors(datatensors_path)

    # Split input data
    _, _, x_train_ens, y_train_ens, x_test, y_test = strat_shuffle_split_training(data, labels)

    # Build base models
    valPerc = Confs.parameters_general["valPerc"]

    #print("Loading base models...")
    base_models = load_models_and_compile(base_models_path, Confs.parameters_general)

    output_file = "results_ensemble_" + combiner_mode + ".txt"

    name_results_file = os.path.join(results_path, output_file)
    f = open(name_results_file, "w")

    all_results_file = Confs.parameters_general['result_path_file']
    f_all_results = open(all_results_file, "a+")
    # ***** PRINT ACCURACY RESULTS: BASE MODELS
    f.write("model\tacc\tauc\tg-mean\tF1\n")
    for model in base_models:
        acc = compute_accuracy(model, x_test, y_test, verbose=2)
        print('Model ' + model.name)
        print('Accuracy ' + str(acc))
        f.write(model.name + '\t')

    # Build ensemble model
    print("Building the " + combiner_mode + " ensemble schema...")
    ensemble_model = build_ensemble_model(x_train_ens, y_train_ens, x_test, y_test, base_models, combiner_mode, results_path, valPerc,
                                          aggressive_opt)

    # reload saved models
    if (Confs.parameters_general["save_and_reload_models_for_accuracy_and_metalearner"]):
        ensemble_model_path = os.path.join(results_path, 'ensemble_model/')
        ensemble_model_filepath = os.path.join(ensemble_model_path, ensemble_model.name + '.h5')
        del ensemble_model
        ensemble_model = load_model_and_compile(ensemble_model_filepath, Confs.parameters_ensemble)

    # ***** PRINT ACCURACY RESULTS: ENSEMBLE
    f.write(ensemble_model.name + '\t')
    f_all_results.write(ensemble_model.name + '\t')
    conf_matrix_file = Confs.parameters_general['conf_matrix_file']
    f_conf_matrix_file = open(conf_matrix_file, "a+")
    f_conf_matrix_file.write(ensemble_model.name + '\n')
    f_conf_matrix_file.close()

    print("\n**********\nENSEMBLE accuracy: model " + ensemble_model.name)
    # make predictions on new text
    y_pred = ensemble_model.predict(x_test)
    score = eval_single_model(y_test, y_pred, cut_off=0.5, decimal_numbers=5)
    print(score)
    print('\n')
    f.write(score)
    f_all_results.write(score)
    f.write("\n")
    f_all_results.write("\n")
    f.close()
    f_all_results.close()


def train_base(dataset_file, results_path, base_models_path=None,
                   text_col_name='text_col_name', label_col_name='label_col_name'):
    csv_separator_char = Confs.parameters_general["csv_separator_char_training"]

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if base_models_path is None:
        base_models_path = os.path.join(results_path, "base_models/")
    datatensors_path = os.path.join(results_path, 'input_tensors/')
    # Read input data
    if os.path.exists(os.path.join(datatensors_path, "data.npy")):
        data, labels = read_training_data_tensors(datatensors_path)
    else:
        extract_train_tensors(dataset_file, datatensors_path, csv_separator_char, text_col_name=text_col_name,
                              label_col_name=label_col_name)
        data, labels = read_training_data_tensors(datatensors_path)
    _, _, x_train_ens, y_train_ens, x_test, y_test = strat_shuffle_split_training(data, labels)

    base_models = load_models_and_compile(base_models_path, Confs.parameters_general)
    all_results_file = Confs.parameters_general['result_path_file']
    f_all_results = open(all_results_file, "a+")
    for model in base_models:
        f_all_results.write(model.name + '\t')
        conf_matrix_file = Confs.parameters_general['conf_matrix_file']
        f_conf_matrix_file = open(conf_matrix_file, "a+")
        f_conf_matrix_file.write(model.name + '\n')
        f_conf_matrix_file.close()
        y_pred = model.predict(x_test)
        score = eval_single_model(y_test, y_pred, cut_off=0.5, decimal_numbers=5)
        f_all_results.write(score)
        f_all_results.write("\n")
    f_all_results.close()

def test(dataset_file, results_path, predictor_path, text_col_name, label_col_name='prediction'):
    csv_separator_char = Confs.parameters_general["csv_separator_char_test"]
    datatensors_path = os.path.join(results_path, 'input_tensors/')
    if ("ensemble" in predictor_path):
        parameters = Confs.parameters_ensemble
    elif ("lstm" in predictor_path):
        parameters = Confs.parameters['lstm']
    elif('cnn' in predictor_path):
        parameters = Confs.parameters['cnn']

    # Read new data
    x_test, texts = extract_test_tensors(dataset_file, datatensors_path, csv_separator_char, text_col_name)

    # Load the prediction model
    model = load_model_and_compile(predictor_path, parameters)

    # make predictions on new text
    pred_labels = model.predict(x_test)

    pred_labels = [pred for pred in pred_labels]

    # build a dataframe with two columns text, pred_label
    data_tuples = list(zip(texts, pred_labels))
    df = pd.DataFrame(data_tuples, columns=[text_col_name, label_col_name])

    # save dataframe on disk as a csv for future consultation
    output_file_name = model.name +"_preds.csv"
    output_file_path = os.path.join(results_path, output_file_name)
    df.to_csv(path_or_buf=output_file_path, index=False, sep='|', encoding=Confs.parameters_general["encoding"])

class TextPreprocessor(object):
    def __init__(self, datatensors_path, transform_single = True):
        self.datatensors_path = datatensors_path
        self.transform_single = transform_single
    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, text):  # comply with scikit-learn transformer requirement
        tensor = extract_tensor(text, self.datatensors_path)
        if self.transform_single:
            tensor = np.array(tensor,)
        return tensor
