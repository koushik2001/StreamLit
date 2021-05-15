import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import nltk
import regex
#nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

data = pd.read_csv(r'Combined_News_DJIA.csv')

data['Top23'].fillna(data['Top23'].median,inplace=True)
data['Top24'].fillna(data['Top24'].median,inplace=True)
data['Top25'].fillna(data['Top25'].median,inplace=True)


 
def create_df(dataset):
    
    dataset = dataset.drop(columns=['Date', 'Label'])
    dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    for col in dataset.columns:
        dataset[col] = dataset[col].str.lower()
        
    headlines = []
    for row in range(0, len(dataset.index)):
        headlines.append(' '.join(str(x) for x in dataset.iloc[row, 0:25]))
        
    df = pd.DataFrame(headlines, columns=['headlines'])
    df['label'] = data.Label
    df['date'] = data.Date
    
    return df

df = create_df(data)

X = df.headlines

def tokenize(text):
    text = regex.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words = 'english',ngram_range=(2,2))),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])


train = df[df['date'] < '20150101']
test = df[df['date'] > '20141231']

x_train = train.headlines
y_train = train.label
x_test = test.headlines
y_test = test.label

#pipeline.fit(x_train, y_train)

#predictions = pipeline.predict(x_test)





import joblib

#joblib.dump(pipeline, 'another_best.pkl')

#load your model for further usage
best_model = joblib.load("another_best.pkl")
#y_im_pred = best_model.predict(x_test)

#print(accuracy_score(y_test,y_im_pred))


def prediction(text):
    labels = ['down','up']

    p = best_model.predict(text)

    s = [str(i) for i in p]

    v = int(''.join(s))

    return str('The stock market will be going '+labels[v])



st.title('Stock Market Analyzer')
st.image('stock.jpg')

user_input = st.text_input('Enter the news')

submit = st.button('Predict')

if(submit):
    answer = prediction([user_input])
    st.text(answer)
