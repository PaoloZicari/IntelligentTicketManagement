import os

from keras import backend as K
from keras.layers import Concatenate, GlobalMaxPool1D, Dense, Multiply, Add, Input, Embedding
from keras.models import Model

import Confs as Confs
from Utils import train, empty_folder

#### ENSEMBLE MODELS ####
# MOE ENSEMBLE
def ensemble_moe(models, ensemble_input, parameters):
    new_models = []
    for model in models:
        model_output = model(ensemble_input)
        new_model = Model(ensemble_input, model_output)
        new_models.append(new_model)

    for i in range(len(new_models)):
        m = new_models[i]
        if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
            freeze_model(m, parameters["freeze_base_models_partly"])

    # Parameters
    params = Confs.parameters['moe']
    vocab_size = Confs.parameters_general['vocab_size']
    embedding_dim = params['embedding_dim']
    dim_dense1 = params["dim_dense1"]
    dim_dense2 = Confs.parameters_general['dim_last_dense']
    activation = Confs.parameters_general['last_activation']
    # Gating network
    input_len = ensemble_input.get_shape().as_list()[1]
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_len, name='embedding')(
        ensemble_input)
    x = GlobalMaxPool1D()(x)
    features = Dense(dim_dense1, activation='relu', kernel_initializer='glorot_normal')(x)
    num_classes = new_models[0].output.shape[1]
    comb_weights_tensor = Dense(num_classes, activation='softmax', kernel_initializer="glorot_normal")(features)
    temp = []
    index = 0
    for model in new_models:
        test_tensor = K.expand_dims(comb_weights_tensor[:, index], axis=1)
        test_tensor = K.repeat_elements(test_tensor, rep=num_classes, axis=1)
        mul_tensor = Multiply()([test_tensor, model.output])
        index = index + 1
        temp.append(mul_tensor)
    output_layer = Add()(temp)
    output_layer = Concatenate()([output_layer, features])
    output_layer = Dense(dim_dense2, activation=activation)(output_layer)
    ensemble_model = Model(ensemble_input, output_layer, name='ensemble_moe')
    ensemble_model.compile(loss=parameters["loss"], optimizer=parameters["optimizer"], metrics=parameters["metrics"])
    return ensemble_model

# STACKING ENSEMBLE
# ENSEMBLE STRATEGY: DEEP STACKING WITH FFNN INPUTTED BY BASE LEARNERS + INPUT FEAURES
def ensemble_stacking_ffnn(models, ensemble_input, parameters):
    new_models = []
    for model in models:
        model_output = model(ensemble_input)
        new_model = Model(ensemble_input, model_output)
        new_models.append(new_model)

    for i in range(len(new_models)):
        m = new_models[i]
        if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
            freeze_model(m, parameters["freeze_base_models_partly"])

    param_nn_gmp = Confs.parameters['ensemble_stacking_ffnn']
    vocab_size = Confs.parameters_general['vocab_size']
    embedding_dim = param_nn_gmp['embedding_dim']
    dim_dense1 = param_nn_gmp["dim_dense1"]
    dim_dense2 = Confs.parameters_general['dim_last_dense']
    activation = Confs.parameters_general['last_activation']

    pred_list = [model.outputs[0] for model in new_models]
    pred_tensor = Concatenate()(pred_list)
    input_len = ensemble_input.get_shape().as_list()[1]
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_len, name='embedding')(ensemble_input)
    x = GlobalMaxPool1D()(x)
    x = Dense(dim_dense1, activation='relu', name='hidden_features_0')(x)
    x = Concatenate()([x, pred_tensor])
    x = Dense(dim_dense2, activation=activation)(x)
    model = Model(ensemble_input, x, name='ensemble_stacking_ffnn')
    loss = Confs.parameters_general['loss']
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.summary()
    return model

def build_ensemble_model(x_train_ens, y_train_ens, x_test, y_test, models, combiner_mode, results_path, valPerc, aggressive_opt=True):
    # ******************************************************************************
    #                   SETTING OF PATHS
    # ******************************************************************************
    ensemble_model_path = os.path.join(results_path, 'ensemble_model/')
    if not os.path.exists(ensemble_model_path):
        os.makedirs(ensemble_model_path)

    parameters = Confs.parameters_ensemble
    valPerc = Confs.parameters_general["valPerc"]

    # ******************************************************************************
    #                           COMPUTATION STARTS HERE
    # ******************************************************************************

    input_shape = x_train_ens.shape[1]
    name_model_input = 'ensemble_input'
    model_input = Input(shape=(input_shape,), dtype='int32', name=name_model_input)

    # ***** LEARN COMBINER
    if combiner_mode == 'moe':
        ensemble_model = ensemble_moe(models, model_input, parameters)
    elif combiner_mode == 'stacking_ffnn':
        ensemble_model = ensemble_stacking_ffnn(models, model_input, parameters)
    else:
        print('No ensemble mode selected')
        exit(0)

    # ensemble_model.summary()
    print("\nComputing the ensemble model - scheme: " + combiner_mode)

    print("Train the ensemble model...")
    _ = train(x_train=x_train_ens, y_train=y_train_ens, val_perc=valPerc,
                  model=ensemble_model, parameters=parameters, filepath=ensemble_model_path,
                  aggressive_opt=aggressive_opt, verbose=2)
    return ensemble_model


def bounded_relu(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced

def freeze_model(model, partly=False):
    if not partly:
        model.trainable = False
    for layer in model.layers:
        if not partly or ("features_0" not in layer.name) and ("features_1" not in layer.name):
            layer.trainable = False