import math
import os
import pickle
import random
import re
import string
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import Confs as Confs

# extract tensors and save them in datatensors_path
def extract_train_tensors(dataset_file, datatensors_path, separator, text_col_name, label_col_name):
    texts, labels = read_training_data(dataset_file, Confs.parameters_general["seed"],label_col_name, text_col_name,
                                       separator)

    texts_clean = clean_text(texts)
    maxlen = Confs.parameters_general["max_len"]

    data, labels, _, tokenizer = preprocessTrainingData(texts_clean, labels, maxlen,
                                                        Confs.parameters_general["max_words"])
    print('Dataset length: ' + str(len(data)))
    # save the input tensors on disk for future re-using
    if not os.path.exists(datatensors_path):
        os.makedirs(datatensors_path)

    np.save(datatensors_path + "data", data)
    np.save(datatensors_path + "labels", labels)

    filehandler = open(datatensors_path + "dictionary.pkl", 'wb')
    pickle.dump(tokenizer, filehandler,  protocol=pickle.HIGHEST_PROTOCOL)
    filehandler.close()


def extract_test_tensors(dataset_file, datatensors_path, separator, text_col_name='SentimentText'):
    texts = read_test_data(dataset_file, text_col_name, separator)

    texts_clean = clean_text(texts)
    maxlen = Confs.parameters_general["max_len"]
    # load the dictionary built during the training phase and reuse it to encode the test set as well
    filehandler = open(datatensors_path + "dictionary.pkl", 'rb')
    tokenizer = pickle.load(filehandler)
    data = preprocessTestData(texts_clean, maxlen, tokenizer)
    return data, texts


# read already saved tensors from folder datatensors_path
def read_training_data_tensors(datatensors_path):
    data = np.load(datatensors_path + "data.npy")
    labels = np.load(datatensors_path + "labels.npy")
    return (data, labels)

# dataset is split_training into a traning and test set (e.g. 90%-10%), and the training set in two subsets: train_base and train_ens
def split_training(data, labels, train_base_perc,split_training_data):
    training_samples = int(math.ceil(Confs.parameters_general["trainPerc"] * data.shape[0]))
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_test = data[training_samples:]
    y_test = labels[training_samples:]
    if split_training_data:
        train_base_samples = int(math.ceil(train_base_perc * x_train.shape[0]))
        x_train_base = x_train[:train_base_samples]
        y_train_base = y_train[:train_base_samples]

        x_train_ens = x_train[train_base_samples:]
        y_train_ens = y_train[train_base_samples:]
    else:
        x_train_base = x_train
        y_train_base = y_train
        x_train_ens = x_train
        y_train_ens = y_train

    return x_train_base, y_train_base, x_train_ens, y_train_ens, x_test, y_test

# dataset is split_training into a traning and test set (e.g. 90%-10%), and the training set in two subsets: train_base and train_ens
def strat_shuffle_split_training(data, labels):
    seed = Confs.parameters_general['strat_shuffle_seed']
    test_size = 1 - Confs.parameters_general["trainPerc"]
    num_split = 1
    stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
    stratifier.get_n_splits(data, labels)
    for train_index, test_index in stratifier.split(data, labels):
        x_train, x_test = np.array([data[i] for i in train_index]),np.array( [data[i] for i in test_index])
        y_train, y_test = np.array([labels[i] for i in train_index]), np.array([labels[i] for i in test_index])
    split_training_data = Confs.parameters_general['split_training_data']
    if split_training_data:
        test_size = 1 - Confs.parameters_general["trainBasePerc"]
        stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
        # stratifier.get_n_splits(data, labels)
        stratifier.get_n_splits(x_train, y_train)
        for train_base_index, train_ens_index in stratifier.split(x_train, y_train):
            x_train_base, x_train_ens = np.array([x_train[i] for i in train_base_index]), np.array([x_train[i] for i in train_ens_index])
            y_train_base, y_train_ens = np.array([y_train[i] for i in train_base_index]), np.array([y_train[i] for i in train_ens_index])
    else:
        x_train_base = x_train
        y_train_base = y_train
        x_train_ens = x_train
        y_train_ens = y_train
    return x_train_base, y_train_base, x_train_ens, y_train_ens, x_test, y_test


# dataset is split_training into a traning and test set (e.g. 90%-10%), and the training set in two subsets: train_base and train_ens
def strat_shuffle_split_training_save_test_indexes(data, labels):
    seed = Confs.parameters_general['strat_shuffle_seed']
    test_size = 1 - Confs.parameters_general["trainPerc"]
    num_split = 1  # ONE SPLIT TRAIN/TEST
    stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
    stratifier.get_n_splits(data, labels)
    for train_index, test_index in stratifier.split(data, labels):
        df = pd.DataFrame(test_index, columns=['test_index'])
        # save dataframe on disk as a csv for future consultation
        df.to_csv(path_or_buf='test_indexes.csv', index=False, sep='|', encoding=Confs.parameters_general["encoding"])

# training set is split in training e validation
def strat_shuffle_split_validation(train_data, train_labels):
    seed = Confs.parameters_general['strat_shuffle_seed']
    test_size = Confs.parameters_general["valPerc"]
    num_split = 1  # ONE SPLIT TRAIN/TEST
    stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
    stratifier.get_n_splits(train_data, train_labels)
    for train_index, val_index in stratifier.split(train_data, train_labels):
        x_train, x_val = np.array([train_data[i] for i in train_index]),np.array( [train_data[i] for i in val_index])
        y_train, y_val = np.array([train_labels[i] for i in train_index]), np.array([train_labels[i] for i in val_index])
    return x_train, y_train, x_val, y_val

# dataset is split_training into a traning and test set (e.g. 90%-10%), and the training set in two subsets: train_base and train_ens
# training set is split in trining and validation
def strat_shuffle_split_training_val(data, labels):
    seed = Confs.parameters_general['strat_shuffle_seed']
    test_size = 1 - Confs.parameters_general["trainPerc"]
    num_split = 1  # ONE SPLIT TRAIN/TEST
    stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
    stratifier.get_n_splits(data, labels)
    for train_index, test_index in stratifier.split(data, labels):
        x_train, x_test = np.array([data[i] for i in train_index]),np.array( [data[i] for i in test_index])
        y_train, y_test = np.array([labels[i] for i in train_index]), np.array([labels[i] for i in test_index])
    split_training_data = Confs.parameters_general['split_training_data']
    if split_training_data:
        test_size = 1 - Confs.parameters_general["trainBasePerc"]
        stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
        stratifier.get_n_splits(x_train, y_train)
        for train_base_index, train_ens_index in stratifier.split(x_train, y_train):
            x_train_base, x_train_ens = np.array([x_train[i] for i in train_base_index]), np.array([x_train[i] for i in train_ens_index])
            y_train_base, y_train_ens = np.array([y_train[i] for i in train_base_index]), np.array([y_train[i] for i in train_ens_index])
    else:
        x_train_base = x_train
        y_train_base = y_train
        x_train_ens = x_train
        y_train_ens = y_train
    # split train_base in train e validation
    test_size = Confs.parameters_general["valPerc"]
    stratifier = StratifiedShuffleSplit(n_splits=num_split, test_size=test_size, random_state=seed)
    stratifier.get_n_splits(x_train_base, y_train_base)
    for train_base_no_val_index, train_base_val_index in stratifier.split(x_train_base, y_train_base):
        x_train_base_no_val, x_train_base_val = np.array([x_train_base[i] for i in train_base_no_val_index]), np.array(
            [x_train_base[i] for i in train_base_val_index])
        y_train_base_no_val, y_train_base_val = np.array([y_train_base[i] for i in train_base_no_val_index]), np.array(
            [y_train_base[i] for i in train_base_val_index])
    x_train_base = x_train_base_no_val
    y_train_base = y_train_base_no_val
    x_val_base = x_train_base_val
    y_val_base = y_train_base_val
    return x_train_base, y_train_base, x_val_base, y_val_base, x_train_ens, y_train_ens, x_test, y_test

# read training dataset from a specific repository
def read_training_data(path_file, seed, label_col_name, text_col_name, separator):
    data = pd.read_csv(path_file, encoding=Confs.parameters_general["encoding"],
                      sep=Confs.parameters_general["csv_separator_char_test"])
    print('class distribution: ')
    print(data[label_col_name].value_counts())
    data = data.sample(frac=1, random_state=seed)
    texts = data[text_col_name].values
    labels = data[label_col_name].values
    num_classes = Confs.parameters_general['num_classes']
    if num_classes > 2:
        labels = keras.utils.to_categorical(labels, num_classes=num_classes)
    return (texts, labels)

# read training dataset from a specific repository
def read_test_data(path_file, text_col_name, separator):
    data = pd.read_csv(path_file, encoding=Confs.parameters_general["encoding"], usecols=[text_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    data.columns = ['text']
    texts = []
    for row in data.iterrows():
        texts.append(row[1]['text'])
    return texts

def calculate_len(texts):
    max = 0
    for ticket in texts:
        cur_len = len(ticket)
        if (cur_len > max):
            max = cur_len
    return max

# build dictionary on training data
def buildDictionary(texts, maxwords):
    tokenizer = Tokenizer(num_words=maxwords, lower=True)
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    Confs.parameters_general['vocab_size'] = vocab_size
    Confs.parameters_ensemble['vocab_size'] = vocab_size
    print('Vocabulary size: ' + str(vocab_size))
    return tokenizer

def preprocessTrainingData(texts, labels, maxlen, maxwords):
    tokenizer = buildDictionary(texts, maxwords)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences,
                         maxlen=maxlen)  # cuts sequences to 1000 and pads them if longer --returns a tensor!!!
    labels = np.asarray(labels)
    return data, labels, word_index, tokenizer

def shuffle_data(data, labels, seed):
    indices = np.arange(data.shape[0])
    random.Random(seed).shuffle(indices)
    data = data[indices]  # mix all data
    labels = labels[indices]  # mix all labels (ricorda che prima erano tutti positivi e poi tutti negativi in ordine)
    return data, labels

def preprocessTestData(texts, maxlen, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences,
                         maxlen=maxlen)  # cuts sequences to 1000 and pads them if longer --returns a tensor!!!
    return data

def extract_tensor(text, datatensors_path):
    #texts_clean = clean_text(text)
    texts_clean = text
    maxlen = Confs.parameters_general["max_len"]
    # load the dictionary built during the training phase and reuse it to encode the test set as well
    filehandler = open(datatensors_path + "dictionary.pkl", 'rb')
    tokenizer = pickle.load(filehandler)
    data = preprocessTestData(texts_clean, maxlen, tokenizer)
    return data

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# Clean tickets from symbols, hyperlinks, hashtag...
def clean_text(texts):
    language = Confs.parameters_general["dataset_language"]
    if (language == "english"):
        stopwords_language = stopwords.words('english')
        stemmer = PorterStemmer()
    elif (language == "italian"):
        stopwords_language = stopwords.words('italian')
        stemmer = SnowballStemmer("italian")  # set the italian stemmer...
    # all emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)
    texts_clean = []
    for ticket in texts:
        # remove stock market tickers like $GE
        ticket = re.sub(r'\$\w*', '', ticket)

        # remove old style text "RT"
        ticket = re.sub(r'^RT[\s]+', '', ticket)

        # remove hyperlinks
        ticket = re.sub(r'https?:\/\/.*[\r\n]*', '', ticket)

        # remove hashtags
        # only removing the hash # sign from the word
        ticket = re.sub(r'#', '', ticket)

        # tokenize tickets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=False)
        ticket_tokens = tokenizer.tokenize(ticket)

        tickets_clean = []
        for word in ticket_tokens:
            if (word not in stopwords_language and  # remove stopwords
                    word not in emoticons and  # remove emoticons
                    word not in string.punctuation):  # remove punctuation

                stem_word = stemmer.stem(word)  # stemming word
                tickets_clean.append(stem_word)
        texts_clean.append(tickets_clean)
    return texts_clean