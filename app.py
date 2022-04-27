from tensorflow.keras.models import Model, load_model
import streamlit as st
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from turtle import color
from keras import backend as K
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


st.set_page_config(page_title = "The Wise Box")

MODEL_PATH = r"/Users/swen/code/swenytheone/analyse/wise_box/version_fake_news/dl_model/model.h5"
MAX_NB_WORDS = 100000 # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 239 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2 # data for validation (not used in training)
EMBEDDING_DIM = 100
tokenizer_file = "tokenizer/tokenizer.pickle"
wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

model_list = ['Linear Regression', 'Gradient Boost Classifier','Decision Tree','RFC Classifier']
model_file_list = [r"/Users/swen/code/swenytheone/analyse/wise_box/version_fake_news/ml_model/LR_model.pkl", r"/Users/swen/code/swenytheone/analyse/wise_box/version_fake_news/ml_model/GBC_model.pkl",r"/Users/swen/code/swenytheone/analyse/wise_box/version_fake_news/ml_model/DT_model.pkl",r"/Users/swen/code/swenytheone/analyse/wise_box/version_fake_news/ml_model/RFC_model.pkl"]


with open(tokenizer_file, 'rb') as handle:
   tokenizer = pickle.load(handle)
@st.cache(allow_output_mutation=True)
def Load_model():    
    model = load_model(MODEL_PATH)
    model.make_predict_function()
    model.summary() # included to make it visible when model is reloaded
    session = K.get_session()
    return model, session
   
@st.cache
def basic_text_cleaning(line_from_column):
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


desc = "This web app detects fake news written in English.\
        You can paste the text directly.\
        This app was developed with the Streamlit, sklearn, tensorflow and Keras libraries.\
        The Github repository of the app is available [here](https://github.com/swenytheone).\
        Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/swensaheb/), [Twitter](https://twitter.com/BeWiseInvestor)\
        or via [e-mail](mailto:swensaheb@gmail.com)."



if __name__ == '__main__':
    
    activities = ['Prediction using DL', 'Prediction using ML']
    
    choice = st.sidebar.selectbox('Choose Prediction Type', activities)
    
    st.title("The Wise Box")
    st.markdown(desc)
    st.write("A fake news classification app utilising DL for classification and 4 traditional ML classifiers")
    
    if choice == "Prediction using DL":
        st.subheader("Input the News content below") 
        sentence = st.text_area("Enter your news content here", "Some news",height=300)
        predict_btt = st.button("predict")
        model, session = Load_model()
        if predict_btt:
            clean_text = []
            K.set_session(session)
            i = basic_text_cleaning(sentence)
            clean_text.append(i)
            sequences = tokenizer.texts_to_sequences(clean_text)
            data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
            prediction = model.predict(data)
            st.text(prediction)
            
            
            st.header("Prediction using LSTM model")

            if prediction[0,0] > 0.5:
                class_label = 'true'
                #st.success('This is not a fake news')
                confidence_rate = (float(prediction[0,0]) - 0.5)*2*100          
            elif prediction[0,0] <= 0.5:
                class_label = 'fake'
                #st.warning('This is a fake news')
                confidence_rate = (0.5 - float(prediction[0,0]))*2*100
            
            
            st.markdown(f" We can say that this news is {class_label} with a **{int(confidence_rate)}%** certainty.")
            if class_label == 'fake':
                #st.image('/Users/swen/code/swenytheone/analyse/fake news/version_fake_news/image/fake_news.jpeg', width = 500)
                st.markdown("![Alt Text](https://media.giphy.com/media/lsOXYUWG04xE1plhXa/giphy.gif)")
                #st.markdown("<h1><span style='color:red'>This is a fake news article!</span></h1>",
                     #unsafe_allow_html=True)

            if class_label == 'true':
                #st.image('/Users/swen/code/swenytheone/analyse/fake news/version_fake_news/image/True_news.jpeg', width = 500)
                st.markdown("![Alt Text](https://media.giphy.com/media/8zT5D0pSGIhyPQ9OdX/giphy.gif)")
            
            
            

                

        
    if choice == "Prediction using ML":
        st.subheader("Input the News content below")    
        sentence = st.text_area("Enter your news content here", "Some news",height=200)
        predict_btt = st.button("predict")
        model, session = Load_model()
        if predict_btt:
            clean_text = []
            K.set_session(session)
            i = basic_text_cleaning(sentence)
            clean_text.append(i)
            sequences = tokenizer.texts_to_sequences(clean_text)
            data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
            prediction = model.predict(data)
            #st.text(prediction)
        
            predictions = []
            for model in model_file_list:
                filename = model
                model = pickle.load(open(filename, "rb"))
                prediction = model.predict([sentence])[0]
                predictions.append(prediction)

            dict_prediction = {"Models":model_list,"predictions":predictions}
            
                 
            df = pd.DataFrame(dict_prediction)

            num_values = df["predictions"].value_counts().tolist()
            num_labels = df["predictions"].value_counts().keys().tolist()

            dict_values = {"true/fake":num_labels,"values":num_values}
            df_prediction = pd.DataFrame(dict_values)
            fig = px.pie(df_prediction, values='values', names='true/fake')
            fig.update_layout(title_text="Comparision between all these 4 models: Prediction proportion between True/Fake")
            st.plotly_chart(fig, use_container_width=True)
            df["predictions"].replace({0: "Fake news", 1: "Reliable news"}, inplace=True)
            st.table(df)
    

        
        
       
        
    