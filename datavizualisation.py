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


data_train = pd.read_json('/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.train.jsonl', lines= True)
data_train.drop(['id'], inplace=True, axis=1)
data_train['new_content'] = data_train['content'].apply(' '.join)
data_train['new_content'][0]

data_train['label'].value_counts()

print(type(data_train['content'][0]))
print(type(data_train['new_content'][0]))
data_train.isnull().sum()


def normalize(data):
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

data_train['new_content'] = normalize(data_train['new_content'])

features = data_train['new_content']
targets = data_train['label']

print(targets.value_counts())
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(targets,ax=ax[0],palette="pastel");
g1.set_title("Count of real and fake data")
g1.set_ylabel("Count")
g1.set_xlabel("Target")
g2 = plt.pie(targets.value_counts().values,explode=[0,0],labels=targets.value_counts().index, autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
fig.show()


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len1=data_train[data_train['label']==0]['new_content'].str.split().map(lambda x: len(x))
ax1.hist(text_len1,color='SkyBlue')
ax1.set_title('Fake news text')
text_len=data_train[data_train['label']==1]['new_content'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='PeachPuff')
ax2.set_title('Real news text')
fig.suptitle('Words in texts')
plt.show()


from statistics import mean
max_len = int(mean(text_len) + mean(text_len1)) / 2 +1


texts = ' '.join(data_train['new_content'])
string = texts.split(" ")


def draw_n_gram(string,i):
    n_gram = (pd.Series(nltk.ngrams(string, i)).value_counts())[:15]
    n_gram_df=pd.DataFrame(n_gram)
    n_gram_df = n_gram_df.reset_index()
    n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
    print(n_gram_df.head())
    plt.figure(figsize = (16,9))
    return sns.barplot(x='count',y='word', data=n_gram_df)


draw_n_gram(string,1)
draw_n_gram(string,2)
draw_n_gram(string,3)
features = data_train['new_content']




