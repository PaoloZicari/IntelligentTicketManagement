####Build different NN-based base classifiers
import os
import numpy as np
from keras.layers import Conv1D, GlobalMaxPooling1D, SpatialDropout1D, BatchNormalization
from keras.layers import Embedding, Flatten, Dense, Dropout, LSTM, GlobalMaxPool1D, MaxPooling1D, GRU
from keras.layers import Input, concatenate
from keras.models import Model
from keras import layers
from keras import backend as K
from tensorflow import keras
import tensorflow as tf

import Confs as Confs
from Utils import train, load_model_and_compile, focal_loss

#### BASE MODELS ####
# TRANSFORMER MODEL
def create_transformer(model_input, pretrainedEmbedding, trainableEmb, model_name):
    maxlen = 30  # Only consider the first 30 words
    vocab_size = 10000  # Only consider the top 10k words
    embed_dim = 32
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    dense_layer_size = 4  # dense layer size
    dropout_pcg = 0.2  # percentage of dropout
    output_dim = 4  # output dim
    inputs = layers.Input(shape=(maxlen,))
    token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    maxlen = tf.shape(inputs)[-1]
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = pos_emb(positions)
    x = token_emb(inputs)
    x = x + positions
    rate = dropout_pcg
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    ffn = keras.Sequential(
        [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
    )
    layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    dropout1 = layers.Dropout(rate)
    dropout2 = layers.Dropout(rate)
    # call
    attn_output = att(x, x)
    attn_output = dropout1(attn_output)
    out1 = layernorm1(x + attn_output)
    ffn_output = ffn(out1)
    ffn_output = dropout2(ffn_output)
    x = layernorm2(out1 + ffn_output)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_pcg)(x)
    x = layers.Dense(dense_layer_size, activation="tanh")(x)
    x = layers.Dropout(dropout_pcg)(x)
    if output_dim == 2:
        outputs = layers.Dense(2, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(output_dim, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    loss = Confs.parameters_general['loss']
    print(loss)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.summary()
    return model

# GRU MODEL
def create_gru(model_input, pretrainedEmbedding, trainableEmb, model_name):  # input_shape,
    parameters = Confs.parameters[model_name]
    vocab_size = Confs.parameters_general['vocab_size']
    embedding_dim = parameters['embedding_dim']
    gru_dim = parameters['lstm_dim']
    input_len = model_input.get_shape().as_list()[1]
    x = Embedding(vocab_size, embedding_dim, input_length=input_len, name='embedding')(model_input)
    x = SpatialDropout1D(0.5)(x)
    x = BatchNormalization()(x)
    x = GRU(gru_dim, recurrent_dropout=0.25, dropout=0.25, name='hidden_features_0')(x)
    x = BatchNormalization()(x)
    dim_dense2 = Confs.parameters_general['dim_last_dense']
    activation = Confs.parameters_general['last_activation']
    x = Dense(dim_dense2, activation=activation)(x)
    model = Model(model_input, x, name=model_name)
    if (pretrainedEmbedding is not None):
        model.layers[1].set_weights([pretrainedEmbedding])
        model.name = 'trained_emb_gru'
        if (trainableEmb):
            model.layers[1].trainable = True
        else:
            model.layers[1].trainable = False
    loss = Confs.parameters_general['loss']
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.summary()
    return model

# LSTM MODEL
def create_lstm(model_input, pretrainedEmbedding, trainableEmb, model_name):  # input_shape,
    parameters = Confs.parameters[model_name]
    vocab_size = Confs.parameters_general['vocab_size']
    embedding_dim = parameters['embedding_dim']
    lstm_dim = parameters['lstm_dim']
    input_len = model_input.get_shape().as_list()[1]
    x = Embedding(vocab_size, embedding_dim, input_length=input_len, name='embedding')(model_input)
    x = SpatialDropout1D(0.5)(x)
    x = BatchNormalization()(x)
    x = LSTM(lstm_dim, recurrent_dropout=0.25, dropout=0.25, name='hidden_features_0')(x)
    x = BatchNormalization()(x)
    dim_dense2 = Confs.parameters_general['dim_last_dense']
    activation = Confs.parameters_general['last_activation']
    x = Dense(dim_dense2, activation=activation)(x)
    model = Model(model_input, x, name=model_name)
    if (pretrainedEmbedding is not None):
        model.layers[1].set_weights([pretrainedEmbedding])
        model.name = 'trained_emb_lstm'
        if (trainableEmb):
            model.layers[1].trainable = True
        else:
            model.layers[1].trainable = False
    loss = Confs.parameters_general['loss']
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.summary()
    return model

# CNN MODEL
def create_cnn(model_input, pretrainedEmbedding, trainableEmb, model_name):  # input_shape,
    parameters = Confs.parameters[model_name]
    vocab_size = Confs.parameters_general['vocab_size']
    embedding_dim = parameters['embedding_dim']
    dim_dense1 = parameters["dim_dense1"]
    n_filters_1 = parameters['n_filters_1']
    dim_filter_1 = parameters['dim_filter_1']
    input_len = model_input.get_shape().as_list()[1]
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_len, name='embedding')(model_input)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(n_filters_1, dim_filter_1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(dim_dense1, activation='relu', name='hidden_features_0')(x)
    dim_dense2 = Confs.parameters_general['dim_last_dense']
    activation = Confs.parameters_general['last_activation']
    x = Dense(dim_dense2, activation=activation)(x)
    model = Model(model_input, x, name=model_name)
    loss = Confs.parameters_general['loss']
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.summary()
    return model

# ******** BUILDING BASE MODEL
def build_base_model(model_name, base_models_path,
                     x_train, y_train, val_perc,
                     pretrained_embedding_filepath, trainableEmb,
                     aggressive_opt=False, verbose=1):

    parameters = Confs.parameters[model_name]
    base_model_file = os.path.join(base_models_path, model_name + ".h5")

    if os.path.exists(base_model_file):
        print("\n***Loading the existing " + model_name + " model***")
        model = load_model_and_compile(base_model_file, parameters)
    else:
        # print("Build ex-novo the base prediction models.")
        print("\n***Building ex-novo the " + model_name + " model***")

        input_shape = x_train.shape[1]
        name_model_input = 'input'
        model_input = createInputModel(input_shape, name_model_input)

        # Build Deep Learning Sentiment Classifiers and look at its accuracy performances on the validation set
        model = eval('create_' + model_name)(model_input, None, trainableEmb, model_name)

        print("\n\n**********************************\nThe chosen model is: " + model.name)
        model.summary()

        # training the neural network...
        train(x_train, y_train, val_perc, model, parameters, base_models_path, aggressive_opt,
              verbose=verbose)  # num_epochs

        if (Confs.parameters_general["save_and_reload_models_for_accuracy_and_metalearner"]):
            del model
            model = load_model_and_compile(base_model_file, parameters)
    return model

# ******** BUILDING BASE MODELS
def build_base_models(x_train_base, y_train_base, x_test, y_test, results_path, model_names, aggressive_opt, valPerc):
    models = []
    pretrained_embedding_filepath = None
    trainableEmb = False  # no fine tuning of the embedding
    base_models_path = os.path.join(results_path, 'base_models/')
    if not os.path.exists(base_models_path):
        os.makedirs(base_models_path)
    for model_name in model_names:
        model = build_base_model(model_name, base_models_path,
                                 x_train_base, y_train_base, valPerc,
                                 pretrained_embedding_filepath, trainableEmb,
                                 aggressive_opt, verbose=2)
        models.append(model)
    return models

# AttributeError: 'Activation' object has no attribute '__name__'
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

def createInputModel(input_shape, name):
    model_input = Input(shape=(input_shape,), dtype='int32', name=name)
    return model_input

def importPretrainedEmbeddings(pretrained_embedding_filepath, word_index, parameters):
    embeddings_index = {}
    f = open(pretrained_embedding_filepath)
    for line in f:
        values = line.split()
        word = values[0]
        embedding = values[1:]
        embeddings_index[word] = embedding
    f.close
    vocab_size = parameters['vocab_size']
    embedding_dim = parameters['embedding_dim']
    weights_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                weights_matrix[i] = embedding_vector
    return weights_matrix