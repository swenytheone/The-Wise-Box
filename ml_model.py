import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_curve,auc
from sklearn.utils import shuffle

import nltk
import nltk as nlp
import string
import re
import pickle

from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm, tqdm_notebook

wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

path = '/content/drive/MyDrive/fake_news_ez'

true = pd.read_csv(path + '/True.csv')
fake = pd.read_csv(path + "/Fake.csv")

true["target"] = 1
fake["target"] = 0

df = pd.concat([true,fake])

df = shuffle(df)
df.head()


true = pd.read_csv(path + '/True.csv')
fake = pd.read_csv(path + "/Fake.csv")

true["target"] = 1
fake["target"] = 0

df = pd.concat([true,fake])

df = shuffle(df)
df.head()


def text_cleaning(line_from_column):
    # This function takes in a string, not a list or an array for the arg line_from_column
    
    tokenized_doc = word_tokenize(line_from_column)
    
    new_review = []
    for token in tokenized_doc:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    new_term_vector = []
    for word in new_review:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    
    final_doc = []
    for word in new_term_vector:
        final_doc.append(wordnet.lemmatize(word))
    
    return ' '.join(final_doc)

def plot_roc_curve(model,y_test, probs, title):
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic - {}'.format(title))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_matrix(classifier,X_test,y_test):
    class_names = df["target"].value_counts()
    np.set_printoptions(precision=2)
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", None)]
    
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)

df["clean_text"] = df["text"].apply(text_cleaning)

def load_classifier(classifier):
    x_train,x_test,y_train,y_test = train_test_split(df['clean_text'], df["target"], test_size=0.25, random_state=2020)
    
    pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', classifier)])
    
    model = pipe.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probs = model.predict_proba(x_test)
    
    plot_matrix(classifier=model,X_test=x_test,y_test=y_test)
    
    return model, probs, y_test

df["clean_text"] = df["text"].apply(text_cleaning)


model_LR, probs, y_test = load_classifier(classifier=LogisticRegression())
plot_roc_curve(model=model_LR,y_test=y_test, probs=probs, title="Logistic Regression")
model_MNB, probs, y_test = load_classifier(classifier=MultinomialNB())
plot_roc_curve(model=model_MNB,y_test=y_test, probs=probs, title = "Multinomial Naive Bayes")
model_BNB, probs, y_test = load_classifier(classifier=BernoulliNB())
plot_roc_curve(model=model_BNB,y_test=y_test, probs=probs, title="Bernoulli Naive Bayes ")
model_GBC, probs, y_test = load_classifier(classifier=GradientBoostingClassifier())
plot_roc_curve(model=model_GBC,y_test=y_test, probs=probs, title="Gradient Boosting Classifier")
model_DT, probs, y_test = load_classifier(classifier=DecisionTreeClassifier())
plot_roc_curve(model=model_DT,y_test=y_test, probs=probs, title="Decision Tree")
model_RFC, probs, y_test = load_classifier(classifier=RandomForestClassifier())
plot_roc_curve(model=model_RFC,y_test=y_test, probs=probs, title="Random Forest Classifier")




# load the model
model_file_list = [path + "/LR_model.pkl",
                   path + "/MNVBC_model.pkl",
                   path + "/BNBC_model.pkl",
                   path + "/GBC_model.pkl",
                   path + "/DT_model.pkl",
                   path + "/RFC_model.pkl"]

model_list = [model_LR,model_MNB,model_BNB,model_GBC,model_DT,model_RFC]

for model, filename in zip(model_list, model_file_list):
    pickle.dump(model, open(filename, 'wb'))