""" On importe les libraries qui nous permettrons de mener à bien le projet """
#Basic libraries
import pandas as pd 
import numpy as np 
import sys, csv, multiprocessing, ujson, keras, re , string
from statistics import mean
from tqdm import tqdm 

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 5]
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

#NLTK libraries
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# Machine Learning libraries
import sklearn 
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier
 
import gensim

#Metrics libraries
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score

#Miscellanous libraries
from collections import Counter

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Deep learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from keras.layers import Input, Conv2D, MaxPool2D, Conv1D, GlobalMaxPooling1D, MaxPool1D
from keras.layers import Reshape, Flatten, Concatenate, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model, Model
from gensim.models.fasttext import FastText
from gensim.parsing.preprocessing import preprocess_string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer


def normalize(data):
    """ une fonction qui permet de retirer d'un text les liens internets, la ponctuation, les caractères
    spéciaux.
    """
    normalized = []
    for i in data:
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

def draw_n_gram(string,i):
    """
    une fonction qui permet de générer un diagramme à barres des n-grammes (séquences de n mots) les plus courants dans un texte donné. 
    """
    n_gram = (pd.Series(nltk.ngrams(string, i)).value_counts())[:15]
    n_gram_df=pd.DataFrame(n_gram)
    n_gram_df = n_gram_df.reset_index()
    n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
    print(n_gram_df.head())
    plt.figure(figsize = (16,9))
    return sns.barplot(x='count',y='word', data=n_gram_df)

def remove_stopwords_and_lemmatization(text):
    """ Une fonction qui permet de Lemmatiser
    pour ramener les formes multiples d'un même mot à leur racine commune
    """
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word) 
            final_text.append(word)
    return " ".join(final_text)

def cleaning(text):
   """ On active la lemmatization sur un texte/date donné """
  
  text = remove_stopwords_and_lemmatization(text)
  return text


@st.cache
def lstm_fake_news():
 """ définit un modèle de mémoire à long terme (LSTM) pour catégoriser les articles de presse comme vrais ou faux """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, output_dim=embed_size, input_length= max_len, trainable = False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 128,  return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(momentum=0.99, trainable = False),
    tf.keras.layers.Dense(units = 64, input_dim=128, activation='relu'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(units = 32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(momentum=0.99, trainable = False),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
    return model
    
def get_pred():
 """ La fonction get_pred évalue le modèle et effectue des prédictions binaires sur la base d'un ensemble de tests donné """
    model.evaluate(X_test, y_test)
    pred = model.predict(X_test)
    binary_predictions = []

    for i in pred:
        if i >= 0.5:
            binary_predictions.append(1)
        else:
            binary_predictions.append(0) 
    return binary_prediction


@st.cache(allow_output_mutation=True)
def load_model(checkpoint_path):
    """La fonction qui nous permet de charger le modèle à partir du meilleur checkpoint calculé
    plus tôt.
    Args:
        checkpoint_path (str): Le chemin du checkpoint à utiliser.
    Returns:
        keras.models.Sequential: Le modèle avec les poids renseignés.
    """
    my_model = lstm_fake_news()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'),
    my_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    my_model.load_weights(checkpoint_path)
    return my_model
