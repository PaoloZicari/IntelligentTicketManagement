########################### SETTINGS  ###########################
# configuration file used for sharing global variables across modules

# PARAMTERS #
parameters = {}
parameters_general = {}

# ENDAVA DATASET SETTINGS #
parameters_general['vocab_size'] = 12011 # size of vocabulary after tokenization of all texts, it is valued after building dictionary
parameters_general["csv_separator_char_training"] = ","
parameters_general["csv_separator_char_test"] = ","
parameters_general['num_classes'] = 4
parameters_general['dataset_folder'] = 'endava'
parameters_general['dataset_name'] = 'dataset/endava/all_tickets.csv'
parameters_general['text_col_name'] = 'body'
parameters_general['label_col_name'] = 'urgency'
parameters_general['class_weight_dict'] = {0: 7.34763514, 1: 1.79862719, 2: 2.19509487, 3: 0.35058676}
# END OF ENDAVA DATASET SETTINGS #

# General settings
parameters_general["max_len"] = 30  # cuts sequences  and add pads them if longer --returns a tensor!!!
parameters_general["max_words"] = 10000  # top 10000 words  max number of words in the dictionary
parameters_general['dim_last_dense'] = 1 # 1 for binary classification - 4 for 4 classes
parameters_general['last_activation'] = 'sigmoid' # 'sigmoid' for binary classification - 'softmax' for 4 classes
parameters_general['do_resample'] = False # resampling of training dataset
parameters_general['class_balancing'] = True # class balancing
parameters_general["dataset_language"] = "english"
parameters_general["csv_separator_char_test"] = ","
parameters_general["encoding"] = "utf-8"
parameters_general["loss"] = 'binary_crossentropy'
parameters_general["optimizer"] = 'adam'
parameters_general["metrics"] = ['acc']
parameters_general["save_and_reload_models_for_accuracy_and_metalearner"] = True
parameters_general['result_path_file'] = 'results.txt'
parameters_general["aggressive_opt"] = True
parameters_general["trainPerc"] = 0.7
parameters_general["valPerc"] = 0.2
parameters_general['split_training_data'] = False
parameters_general['strat_shuffle_seed'] = 24
parameters_general["trainBasePerc"] = 0.8
parameters_general["seed"] = 123
parameters_general["use_tf_data"] = False
parameters_general["load_data_from_binary_files"] = False
parameters_general["max_docs_per_class"] = None
parameters_general["debug"] = True
parameters_general["conf_matrix_file"] = 'conf_matrix_file.txt'

# Base models settings
# lstm parameter dictionary
parameters['lstm'] = {}
parameters['lstm'].update(parameters_general)
parameters['lstm']['vocab_size'] = parameters_general["max_words"]
parameters['lstm']["embedding_dim"] = 128
parameters['lstm']["lstm_dim"] = 256
parameters['lstm']["num_epochs"] = 15
parameters['lstm']["batch_size"] = 128

# GRU parameter dictionary
parameters['gru'] = {}
parameters['gru'].update(parameters_general)
parameters['gru']['vocab_size'] = parameters_general["max_words"]
parameters['gru']["embedding_dim"] = 128
parameters['gru']["lstm_dim"] = 256
parameters['gru']["num_epochs"] = 15
parameters['gru']["batch_size"] = 128

# CNN parameter dictionary
parameters['cnn'] = {}
parameters['cnn'].update(parameters_general)
parameters['cnn']['vocab_size'] = parameters_general["vocab_size"]
parameters['cnn']["embedding_dim"] = 128
parameters['cnn']["n_filters_1"] = 16
parameters['cnn']["dim_filter_1"] = 5
parameters['cnn']["dim_dense1"] = 64
parameters['cnn']["dim_dense2"] = 1
parameters['cnn']["num_epochs"] = 32
parameters['cnn']["batch_size"] = 64

# transformer parameter dictionary
parameters['transformer'] = {}
parameters['transformer'].update(parameters_general)
parameters['transformer']['vocab_size'] = parameters_general["vocab_size"]
parameters['transformer']["embedding_dim"] = 32
parameters['transformer']["num_epochs"] = 10
parameters['transformer']["batch_size"] = 128

# Ensemble settings
# ensemble parameter dictionary
parameters_ensemble = {}
parameters_ensemble.update(parameters_general)
parameters_ensemble["freeze_base_models"] = True
parameters_ensemble["freeze_base_models_partly"] = False
parameters_ensemble["recompute_extended_base_models"] = True
parameters_ensemble["sse_averaging_per_base_model"] = False
parameters_ensemble["num_epochs"] = 25
parameters_ensemble["batch_size"] = 128
parameters_ensemble["loss"] = 'binary_crossentropy'
parameters_ensemble["optimizer"] = 'adam'
parameters_ensemble["metrics"] = ['acc']
parameters_ensemble["vocab_size"] = parameters_general["vocab_size"]
parameters_ensemble["embedding_dim"] = 128

# MOE parameter dictionary
parameters['moe'] = {}
parameters['moe'].update(parameters_general)
parameters['moe']['vocab_size'] = parameters_general["vocab_size"]
parameters['moe']["embedding_dim"] = 128
parameters['moe']["dim_dense1"] = 64

# ensemble_stacking_ffnn parameter dictionary
parameters['ensemble_stacking_ffnn'] = {}
parameters['ensemble_stacking_ffnn'].update(parameters_general)
parameters['ensemble_stacking_ffnn']['vocab_size'] = parameters_general["vocab_size"]
parameters['ensemble_stacking_ffnn']["embedding_dim"] = 128
parameters['ensemble_stacking_ffnn']["dim_dense1"] = 64
parameters['ensemble_stacking_ffnn']["dim_dense2"] = 1
parameters['ensemble_stacking_ffnn']["num_epochs"] = 32
parameters['ensemble_stacking_ffnn']["batch_size"] = 64
###############################################################################