import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from scipy.spatial.distance import cdist
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
# Download stopwords list
nltk.download('punkt')
from gensim.parsing.preprocessing import remove_stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

import Confs as Confs
from BuildInputTensors import strat_shuffle_split_training, extract_test_tensors, clean_text, extract_tensor
from TrainTest_Interface import TextPreprocessor

def create_testset_trainset(dataset_file, results_path, text_col_name, label_col_name, trainset_file, testset_file):
    csv_separator_char = Confs.parameters_general["csv_separator_char_test"]
    # Read new data
    #x_test, texts = extract_test_tensors(dataset_file, datatensors_path, csv_separator_char, text_col_name)
    # labels
    label = pd.read_csv(dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[label_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    text = pd.read_csv(dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[text_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])

    label.columns = [label_col_name]
    labels = []
    for row in label.iterrows():
        labels.append(row[1][label_col_name])

    text.columns = [text_col_name]
    texts = []
    for row in text.iterrows():
        texts.append(row[1][text_col_name])

    # Split input data
    x_train_base, y_train_base, _, _, x_test, y_test = strat_shuffle_split_training(texts, labels)

    # testset
    data_tuples_test = list(zip(x_test, y_test))
    df = pd.DataFrame(data_tuples_test, columns=[text_col_name, label_col_name])
    # save
    output_file_name = testset_file
    output_file_path = os.path.join(results_path, output_file_name)
    df.to_csv(path_or_buf=output_file_path, index=False, sep=csv_separator_char, encoding=Confs.parameters_general["encoding"])

    # trainset
    data_tuples_train = list(zip(x_train_base, y_train_base))
    df = pd.DataFrame(data_tuples_train, columns=[text_col_name, label_col_name])
    # save
    output_file_name = trainset_file
    output_file_path = os.path.join(results_path, output_file_name)
    print(output_file_path)
    df.to_csv(path_or_buf=output_file_path, index=False, sep=csv_separator_char, encoding=Confs.parameters_general["encoding"])


def create_lat_rep(dataset_file, results_path, predictor_path, text_col_name, out_file_name, label_col_name, lat_rep_col_name='predicted_lat_rep'):
    csv_separator_char = Confs.parameters_general["csv_separator_char_test"]
    datatensors_path = os.path.join(results_path, 'input_tensors/')
    parameters = Confs.parameters_ensemble
    # Load the prediction model
    model = load_model(predictor_path)
    model.compile(loss=parameters["loss"], optimizer=parameters["optimizer"], metrics=parameters["metrics"])
    input_model = model.input
    output_model = model.layers[len(model.layers)-2].output
    model_test = Model(input_model, output_model, name='model_test')
    model_test.compile(loss=parameters["loss"], optimizer=parameters["optimizer"], metrics=parameters["metrics"])

    # Read  dataset
    x_test, texts = extract_test_tensors(dataset_file, datatensors_path, csv_separator_char, text_col_name)
    data = pd.read_csv(dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[label_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    data.columns = [label_col_name]
    labels = []
    for row in data.iterrows():
        labels.append(row[1][label_col_name])

    # create latent representation from model_test (i.e. model without last layer) predictions make predictions
    lat_reps = model_test.predict(x_test)
    lat_reps = [pred for pred in lat_reps]

    #make predictions and probabilities
    predictions = model.predict(x_test)
    pred_labels = [np.argmax(pred) for pred in predictions]
    probabilities = [pred for pred in predictions]

    # build a dataframe with two columns text, pred_label
    data_tuples = list(zip(texts, labels, pred_labels, probabilities, lat_reps))
    df = pd.DataFrame(data_tuples, columns=[text_col_name, label_col_name, 'prediction', 'probabilities', lat_rep_col_name])

    # save dataframe on disk as a csv for future consultation
    output_file_path = os.path.join(results_path, out_file_name)
    df.to_csv(path_or_buf=output_file_path, index=False, sep=csv_separator_char, encoding=Confs.parameters_general["encoding"])


def find_k_neighbours(train_dataset_file, test_dataset_file, results_path, selected_tuple, n_neighbors, text_col_name, label_col_name, output_file_name, lat_rep_col_name='predicted_lat_rep'):
    train_dataset_file = os.path.join(results_path, train_dataset_file)
    test_dataset_file = os.path.join(results_path, test_dataset_file)
    csv_separator_char = Confs.parameters_general["csv_separator_char_test"]
    test_df = pd.read_csv(test_dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[text_col_name , lat_rep_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])

    print('Text of the target ticket:')
    print(test_df[text_col_name][selected_tuple])
    target_lat_rep = test_df[lat_rep_col_name][selected_tuple].strip('[]').split()
    print('Representation in the latent space of the target ticket')
    print(target_lat_rep)
    data = pd.read_csv(train_dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[lat_rep_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    data.columns = [lat_rep_col_name]
    train_lat_reps = []
    for row in data.iterrows():
        train_lat_reps.append(row[1][lat_rep_col_name].strip('[]').split())

    label = pd.read_csv(train_dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[label_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    label.columns = [label_col_name]
    labels = []
    for row in label.iterrows():
        labels.append(row[1][label_col_name])

    prediction_col_name = 'prediction'
    prediction = pd.read_csv(train_dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[prediction_col_name],
                        sep=Confs.parameters_general['csv_separator_char_test'])
    prediction.columns = [prediction_col_name]
    predictions = []
    for row in prediction.iterrows():
        predictions.append(row[1][prediction_col_name])

    text = pd.read_csv(train_dataset_file, encoding=Confs.parameters_general["encoding"], usecols=[text_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    text.columns = [text_col_name]
    texts = []
    for row in text.iterrows():
        texts.append(row[1][text_col_name])

    # find k neighbours
    distances = cdist([target_lat_rep], train_lat_reps, metric='euclidean')
    # cdist with cosine computes 1 - cos(theta)
    #distances = cdist([target_lat_rep], train_lat_reps, metric='cosine')

    dist = []
    neigh_ind = []
    for row in distances:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:n_neighbors]
        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]
        dist.append(dist_list)
        neigh_ind.append(ind_list)
    #print('nearest_neighbour_indexes: ')
    #print(neigh_ind)
    #print('nearest_neighbour_distances: ')
    #print(dist_list)
    nearest_neighbours = [train_lat_reps[i] for i in ind_list]
    #print('nearest_neighbours: ')
    #print(nearest_neighbours)
    nearest_neighbours_texts = [texts[i] for i in ind_list]
    #print('nearest_neighbours_texts: ')
    #print(nearest_neighbours_texts)

    nearest_neighbours_labels = [labels[i] for i in ind_list]
    nearest_neighbours_predictions = [predictions[i] for i in ind_list]

    #print('nearest_neighbours_labels: ')
    #print(nearest_neighbours_labels)
    #print('nearest_neighbours_predictions: ')
    #print(nearest_neighbours_predictions)

    # save file
    # build a dataframe
    data_tuples = list(zip(ind_list, nearest_neighbours_labels, nearest_neighbours_predictions, dist_list, nearest_neighbours_texts, nearest_neighbours))
    df = pd.DataFrame(data_tuples, columns=['index', label_col_name, prediction_col_name, 'distance', text_col_name, lat_rep_col_name])

    # save dataframe on disk as a csv for future consultation
    output_file_path = os.path.join(results_path, output_file_name)
    df.to_csv(path_or_buf=output_file_path, index=False, sep=csv_separator_char,
              encoding=Confs.parameters_general["encoding"])


def limeExplain(testset_file, id_target, predictor_name, text_col_name, label_col_name, results_path):
    testset_file = os.path.join(results_path, testset_file)
    predictor_path = os.path.join(results_path, predictor_name)
    parameters = Confs.parameters_ensemble
    # Load the prediction model
    model = load_model(predictor_path)
    model.compile(loss=parameters["loss"], optimizer=parameters["optimizer"], metrics=parameters["metrics"])
    label_test = pd.read_csv(testset_file, encoding=Confs.parameters_general["encoding"], usecols=[label_col_name],
                        sep=Confs.parameters_general['csv_separator_char_test'])
    text = pd.read_csv(testset_file, encoding=Confs.parameters_general["encoding"], usecols=[text_col_name],
                       sep=Confs.parameters_general['csv_separator_char_test'])
    label_test.columns = ['label']
    labels_test = []
    for row in label_test.iterrows():
        labels_test.append(row[1]['label'])
    text.columns = [text_col_name]
    texts = []
    for row in text.iterrows():
        texts.append(row[1][text_col_name])

    #preprocess text
    texts = clean_text(texts)
    class_names = range(0, len(label_test['label'].unique()))
    explainer = LimeTextExplainer(class_names=class_names)
    print('Text of the target ticket')
    print(texts[id_target])
    datatensors_path = os.path.join(results_path, 'input_tensors/')
    clf = model
    preprocessor = TextPreprocessor(datatensors_path = datatensors_path)
    # just create the pipeline
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])

    single_text = " "
    for ele in texts[id_target]:
        single_text = single_text + " " + ele

    exp = explainer.explain_instance(single_text, pipeline.predict, num_features=6, labels=[0, 1, 2, 3])
    print('Document id: %d' % id_target)
    print('True class: %s' % class_names[labels_test[id_target]])

    class_index = 0
    print('LIME Explanation for class %s' % class_names[class_index])
    print('\n'.join(map(str, exp.as_list(label=class_index))))

    class_index = 1
    print('LIME Explanation for class %s' % class_names[class_index])
    print('\n'.join(map(str, exp.as_list(label=class_index))))

    class_index = 2
    print('LIME Explanation for class %s' % class_names[class_index])
    print('\n'.join(map(str, exp.as_list(label=class_index))))

    class_index = 3
    print('LIME Explanation for class %s' % class_names[class_index])
    print('\n'.join(map(str, exp.as_list(label=class_index))))

    exp.show_in_notebook(text=False)
    exp.save_to_file(os.path.join(results_path, 'lime.html'))


# wordcloud
def wordCloudsExplain(knn_file_name, label_col_name, text_col_name, results_path, n):
    #print('wordCloudsExplain')
    # read the csv of the k neighbours
    knn_file_name = os.path.join(results_path, knn_file_name)
    df_k_neighbours = pd.read_csv(knn_file_name, encoding=Confs.parameters_general["encoding"],
                        usecols=[text_col_name, label_col_name],sep=Confs.parameters_general['csv_separator_char_test'])
    # select only the n texts of each class
    df_k_neighbours_target_class0 = df_k_neighbours[df_k_neighbours[label_col_name]==0].head(n)
    df_k_neighbours_target_class1 = df_k_neighbours[df_k_neighbours[label_col_name] == 1].head(n)
    df_k_neighbours_target_class2 = df_k_neighbours[df_k_neighbours[label_col_name] == 2].head(n)
    df_k_neighbours_target_class3 = df_k_neighbours[df_k_neighbours[label_col_name] == 3].head(n)
    all_stop_words = stopwords.words('english')
    all_stop_words.extend(sw_list)
    stop_words = set(all_stop_words)
    # Create WordNetLemmatizer object
    wnl = WordNetLemmatizer()
    all_texts = []
    for row in df_k_neighbours.iterrows():
        all_texts.append(remove_stopwords(row[1][text_col_name]))
    # preprocess with stemming
    all_texts = clean_text(all_texts)
    all_texts_without_sw = []
    for sentence in all_texts:
        words = sentence
        txt = [wnl.lemmatize(word) for word in words if not word in all_stop_words]
        all_texts_without_sw.append(' '.join(txt))
    all_texts = all_texts_without_sw

    # Lemmatize the stop words
    tokenizer = LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words))
    vectorizer_all = TfidfVectorizer(stop_words=token_stop, max_df=0.95, min_df=0.05, max_features=400)
    X_all = vectorizer_all.fit_transform(all_texts)
    cl0_texts = processText(df_k_neighbours_target_class0, text_col_name, wnl, all_stop_words)
    cl1_texts = processText(df_k_neighbours_target_class1, text_col_name, wnl, all_stop_words)
    cl2_texts = processText(df_k_neighbours_target_class2, text_col_name, wnl, all_stop_words)
    cl3_texts = processText(df_k_neighbours_target_class3, text_col_name, wnl, all_stop_words)
    createWordCloud('all', all_texts, vectorizer_all, results_path)
    createWordCloud(0, cl0_texts, vectorizer_all, results_path)
    createWordCloud(1, cl1_texts, vectorizer_all, results_path)
    createWordCloud(2, cl2_texts, vectorizer_all, results_path)
    createWordCloud(3, cl3_texts, vectorizer_all, results_path)

def processText(df,text_col_name, wnl, all_stop_words):
    texts = []
    for row in df.iterrows():
        texts.append(remove_stopwords(row[1][text_col_name]))
    texts = clean_text(texts)
    texts_without_sw = []
    for sentence in texts:
        words = sentence
        txt = [wnl.lemmatize(word) for word in words if not word in all_stop_words]
        texts_without_sw.append(' '.join(txt))
    class_texts = texts_without_sw
    return class_texts

def createWordCloud(nclass, texts, vectorizer, results_path):
    str = " "
    for ele in texts:
        str += ele
    wordlist = str.split()
    wordfreq = []
    for p in wordlist:
       if (p in vectorizer.vocabulary_):
            wordfreq.append(wordlist.count(p) * vectorizer.idf_[vectorizer.vocabulary_[p]])
       else:
            wordfreq.append(0)
    word_dict = dict(list(zip(wordlist, wordfreq)))
    if(len(word_dict)==0):
        print("No neighbors in class " , nclass)
    else:
        # wordcloud count of term in all documents * idf (senza log)
        wordcloudgenerator = WordCloud(width=480, height=480, max_words=40, background_color="white")
        wordcloud = wordcloudgenerator.generate_from_frequencies(word_dict)
        # plot the WordCloud image
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.margins(x=0, y=0)
        output_file = f"WordCloud_cl_{nclass}.png"
        output_file = os.path.join(results_path, output_file)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

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


sw_list = ['hello', 'please', 'thank', 'thanks', 'regards', 'kind', 'hi', 'pm', 'has', #'high', 'low',
               'date', 'day', 'days',
               'best', 'guy', 'let', 'dc',
               'month',
               'got', 'old',
               'add', 'dear', 'ad',
               'en', 'able', 'ext',
               'th', 'try',
               'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
               'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
               'november', 'december', 'februari', 'juli',
                'tri'
           ]

# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]
