import pandas as pd
%matplotlib inline
%load_ext autoreload
import multiprocess
import sys
import csv
import multiprocessing
from sklearn.utils import shuffle
from p_tqdm import p_map

import ujson
import keras
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Conv1D, GlobalMaxPooling1D, MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model, Model
import tensorflow as tf
import gensim
from gensim.models.fasttext import FastText
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
from tqdm import tqdm 
import seaborn as sns
import numpy as np

csv.field_size_limit(500 * 1024 * 1024)

path = '/Users/swen/Downloads/FakeNewsCorpus_1_0/'

path_news_csv = path + 'news_cleaned_2018_02_13-1.csv'

path = '/content/drive/MyDrive/ColabNotebooks/'
path_news_preprocessed = path + 'news_cleaned_2018_02_13.preprocessed.jsonl'
path_news_shuffled = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.jsonl'
path_news_train = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.train.jsonl'
path_news_test = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.test.jsonl'
path_news_val = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.val.jsonl'
path_news_embedded =  path + 'news_cleaned_2018_02_13.embedded.jsonl'

for df_news_chunk in pd.read_csv(path_news_csv, chunksize=100000):
    df_news_chunk = shuffle(df_news_chunk)
    print(df_news_chunk)
    break

df_news_chunk.columns

def news_generator():
    with tqdm() as progress:
        for df_news_chunk in pd.read_csv(path_news_csv, encoding='utf-8', engine='python', chunksize=10 * 1000):
            news_filter = df_news_chunk.type.isin(set(['fake', 'conspiracy', 'unreliable', 'reliable']))
            df_news_chunk_filtered = df_news_chunk[news_filter]
            for row in df_news_chunk_filtered.itertuples():
                label = 1 if row.type == 'reliable' else 0

                progress.update()
                yield int(row.id), '%s %s' % (row.title, row.content), label
                
lens = []
for i, (_id, con, label) in enumerate(news_generator()):
    if i > 10 * 1000:
        break

    lens.append(len(con))
    
with tqdm() as progress:
    
    for k in range(5000000):
        progress.update()


# changement ici de multiprocessing en multiprocess
def _preprocess_string(news):
    _id, con, label = news
    return _id, preprocess_string(con), label

def news_preprocessed_generator():
    missing_words = {}
    
    with multiprocess.Pool(multiprocess.cpu_count(), maxtasksperchild=1) as pool:
        for _id, con, label in pool.imap(_preprocess_string, news_generator(), chunksize=1000):
            yield _id, con, label, missing_words

all_missing_words = {}
with open(path_news_preprocessed, 'w') as out_news_embedded:
    for _id, con, label, missing_words in news_preprocessed_generator():
        out_news_embedded.write(ujson.dumps({
            'id': _id, 'content': con, 'label': int(label)
        }) + '\n')
        all_missing_words.update(missing_words)

!shuf /content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.jsonl > \
      /content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.jsonl
      
      
count_lines = 0
with open(path_news_shuffled, 'r') as in_news:
    for line in tqdm(in_news):
        count_lines += 1
        
        
count_lines, int(count_lines * .8), int(count_lines * .1), \
    count_lines - (int(count_lines * 0.8) + int(count_lines * 0.1))
    
subdataset_size = int(count_lines * .05)

with open(path_news_shuffled, 'r') as in_news:
    with open(path_news_train, 'w') as out_train:
        with open(path_news_test, 'w') as out_test:
            with open(path_news_val, 'w') as out_val:
                for i, line in tqdm(enumerate(in_news)):
                    if i < count_lines * .1:
                        out_train.write(line)
                    #elif i < count_lines * .2:
                     #   out_test.write(line)
                    #else:
                    #   out_val.write(line)
                    
