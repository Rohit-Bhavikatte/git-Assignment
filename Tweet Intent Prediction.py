# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:08:04 2022

@author: ROHIT
"""

import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


dataset = pd.read_csv(r"D:\downloads\ASSIGNMENTS\Null Inovation/Tweet_NFT.xlsx - Sheet1.csv")
dataset.head()


dataset.shape
dataset.isnull().sum()
dataset.info()


# Droping unnecessary variables
dataset = dataset.drop(["tweet_created_at","id"],axis=1)


dataset.head()

dataset['tweet_intent'].value_counts()

dataset['tweet_intent'].isnull().sum()
dataset['tweet_intent'].value_counts()

dataset['tweet_intent'].values
dataset[dataset['tweet_intent']=="Launching Soon"] = "Launching_soon"

dataset.isnull().sum()


### We have to clean text and remove unnecessary part  by using NLTK 
### The below code will perform Some Basic NLTK task on our tweet text


text = dataset.iloc[:,0].values
wordnet_lemmatizer = WordNetLemmatizer()
stops = stopwords.words('english') 
nonan = re.compile(r'[^a-zA-Z ]')
clean_X = [] 
for i in range(len(text)):
    sentence = nonan.sub('', text[i])
    words = word_tokenize(sentence.lower())
    filtered_words = [w for w in words if not w.isdigit() and not w in stops and not w in string.punctuation]
    tags = pos_tag(filtered_words)
    cleaned = ''
    pos = ['NN','NNS','NNP','NNPS','RP','MD','FW','VBZ','VBD','VBG','VBN','VBP','RBR','JJ','RB','RBS','PDT','JJ','JJR','JJS','TO','VB']
    for word, tag in tags:
        if tag in pos:
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
       
    clean_X.append(cleaned)
    

X = pd.DataFrame(clean_X)




### We have to divide our data into dependent and independent variables and splitting it into training and testing.



dataset.iloc[96363:96370]
X_train = X.iloc[:96364].values
X_train


X_test = X.iloc[96364:].values
X_test

y_train = dataset.iloc[:96364,1].values
y_train

# We are using TF/IDF Vectorizer methiod




tfidf_vectorize = TfidfVectorizer()
tfidf_vectorize.fit(X_train.ravel())
  

X_train = tfidf_vectorize.fit_transform(X_train.ravel())
X_test = tfidf_vectorize.fit_transform(X_test.ravel())




# The dependent varable contains multiple categorical classes for this we are going to encode it by using Label Encoder
# As i am using LabelEncoder because the variale that are categorical are the target variable

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)


# it refers the labelles classes
encoder.classes_

### Here Appreciation = 0, Community = 1, Done = 2, Giveaway = 3, Interested = 4 , Launching_soon = 5, Presale = 6 ,Whitelist = 7, pinksale =8



# Preparing model
# I am Choosing XG Boost model

classifier = XGBClassifier()
classifier.fit(X_train,y_train)


# For Testing Purpose i am predecting the training data after that we will see our model performane

y_pred = classifier.predict(X_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy is :", accuracy_score(y_train, y_pred))
print("COnfusion Matrix : ", confusion_matrix(y_train, y_pred))
print("Classification Report : ", classification_report(y_train, y_pred))



# From Above we can see that Our model got 97% of accuracy that is a best model



