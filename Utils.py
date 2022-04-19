import glob
import os

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_auc_score
from Attention import Attention
from BuildInputTensors import strat_shuffle_split_validation
import Confs as Confs

def train(x_train, y_train, val_perc, model, parameters, filepath,
          aggressive_opt=False, save_weights_only=False, verbose=0):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    batch_size = parameters["batch_size"]
    file_weights_name = os.path.join(filepath, model.name + '.h5')

    # save the whole model
    checkpoint = ModelCheckpoint(file_weights_name, monitor='val_loss', verbose=2,
                                 save_weights_only=save_weights_only,
                                 save_best_only=True, mode='min')
    if aggressive_opt:
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=6, min_delta=0.0001, verbose=verbose,
                                       mode='auto', cooldown=0, min_lr=0)
        callbacks = [checkpoint, lr_reducer]
    else:
        callbacks = [checkpoint]
    x_growing, y_growing, x_validation, y_validation = strat_shuffle_split_validation(x_train, y_train)
    y_grow = np.argmax(y_growing, axis=1)
    random_state_vec = [42, 153, 23, 97]
    if Confs.parameters_general['do_resample']:
        print(len(y_growing))
        num_class_0 = len(y_grow[y_grow==0])
        print(num_class_0)
        num_class_1 = len(y_grow[y_grow == 1])
        num_class_2 = len(y_grow[y_grow == 2])
        num_class_3_over = num_class_0 + num_class_1 + num_class_2
        down__sampling_strategy = {0: num_class_0, 1: num_class_1,
                                   2: num_class_2, 3: num_class_3_over}
        if "weight0" in model.name:
            over_sampling_strategy = {0: num_class_3_over, 1: num_class_1, 2: num_class_2, 3: num_class_3_over}
            seed = random_state_vec[0]
        elif "weight1" in model.name:
            over_sampling_strategy = {0: num_class_0, 1: num_class_3_over, 2: num_class_2, 3: num_class_3_over}
            seed = random_state_vec[1]
        elif "weight2" in model.name:
            over_sampling_strategy = {0: num_class_0, 1: num_class_1, 2: num_class_3_over, 3: num_class_3_over}
            seed = random_state_vec[2]
        else:
            over_sampling_strategy = {0: num_class_0, 1: num_class_1, 2: num_class_2, 3: num_class_3_over}
            seed = random_state_vec[3]
        under = RandomUnderSampler(sampling_strategy=down__sampling_strategy, random_state=seed)
        X_under, y_under = under.fit_resample(x_growing, y_growing)
        over = RandomOverSampler(sampling_strategy=over_sampling_strategy, random_state=seed)
        X_over, y_over = over.fit_resample(X_under, y_under)

        x_growing = X_over
        y_growing = y_over
    # weights used to balance the classes distribution
    class_weight_dict = Confs.parameters_general['class_weight_dict']
    num_epochs = parameters["num_epochs"]
    # weight for loss
    if Confs.parameters_general['class_balancing']:
        # BALANCING
        history = model.fit(x_growing, y_growing, batch_size=batch_size, epochs=num_epochs, verbose=verbose,
                           callbacks=callbacks, validation_data=(x_validation, y_validation), class_weight=class_weight_dict)
    else:
        # NO BALANCING
        history = model.fit(x_growing, y_growing, batch_size=batch_size, epochs=num_epochs, verbose=verbose,
                               callbacks=callbacks, validation_data=(x_validation, y_validation))
    return history

def evaluate_error(model, x_test, y_test, verbose=2):
    pred = model.predict(x_test, batch_size=32, verbose=verbose)
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error

def compute_accuracy(model, x_test, y_test, verbose=2):
    _, acc = model.evaluate(x_test, y_test, verbose=verbose)
    return acc

def compute_AUC(model, x_test, y_test, verbose=2):
    # print("Computing prediction probabilities....")
    y_hat = model.predict(x_test, verbose=verbose)
    if Confs.parameters_general['num_classes'] == 2:
        pct_auc = roc_auc_score(y_test, y_hat) * 100.0
    else:
        pct_auc = roc_auc_score(y_test, y_hat, average='weighted', multi_class='ovo') * 100.0
    return pct_auc

# load the weights obtained for the best training iteration
def load_model_and_compile(filepath_model, parameters):
    if ("ensemble" in filepath_model or "bi_lstm" in filepath_model):
        model = load_model(filepath_model, custom_objects={'Attention': Attention})
    else:
        model = load_model(filepath_model, custom_objects={'weighted_loss_tf':weighted_loss_tf})
    model.compile(loss=parameters["loss"], optimizer=parameters["optimizer"], metrics=parameters["metrics"])
    return model

# load the weights obtained for the best training iteration
def load_models_and_compile(filepath_models, parameters):
    if not os.path.exists(filepath_models):
        return []
    else:
        files = []
        for r, d, f in os.walk(filepath_models):
            for file in f:
                if '.h5' in file or '.hdf5' in file:
                    files.append(os.path.join(r, file))
        models = []
        for file in files:
            if ('bi_lstm' in file):
                model = load_model(file, custom_objects={'Attention': Attention})
            else:
                model = load_model(file)

            model.compile(loss=parameters["loss"], optimizer=parameters["optimizer"], metrics=parameters["metrics"])
            models.append(model)
        return models

def empty_folder(folder_path):
    if os.path.exists(folder_path):
        print("Deleting files in folder " + folder_path)
        files = glob.glob(folder_path + '/*')
        for f in files:
            os.remove(f)

# weighted MAE
def weighted_loss_tf(y_true, y_pred):
    weight = 4.
    w = weight * (y_true + 1)
    return K.mean(K.abs(y_pred - y_true) * w)

def focal_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 4
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)
    # Calculate Focal Loss
    loss = alpha*K.pow(1 - y_pred, gamma)*cross_entropy
    # Compute mean loss in mini_batch
    return K.mean(loss, axis=1)

get_custom_objects().update({"focal_loss": focal_loss})
get_custom_objects().update({"weighted_loss_tf": weighted_loss_tf})

