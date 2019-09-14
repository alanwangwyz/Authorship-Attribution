import re, pickle, os, string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np 
import pandas as pd 
import nltk
import string
import spacy
from nltk.corpus import stopwords
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import csv

train_data = None
test_data = None

def load_data():
    global train_data, test_data
    train_data = pd.read_csv('train_tweets.txt', delimiter="\t", header = None, quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv('test_tweets_unlabeled.txt', delimiter="\t", header = None, quoting=csv.QUOTE_NONE)
    
load_data()
train_data.columns = ['label','Tweet']
test_data.columns = ['Tweet']

print(train_data)
print(test_data)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 200000)
allData = train_data['Tweet'].values.tolist() + test_data['Tweet'].values.tolist()
bowAllData = vectorizer.fit_transform(allData) 
train_tagged = vectorizer.get_feature_names()

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

lentrain = len(train_data)

# Separate back into training and test sets.
train = bowAllData[:lentrain]  
test = bowAllData[lentrain:]

lsvc=LinearSVC()
lsvc.fit(train, train_data['label']) # training the model

prediction = lsvc.predict(test)

test_data['Predicted'] = prediction
submission = test_data[['Predicted']]
submission.index = np.arange(1, len(submission) + 1)
submission['Id'] = submission.index

columnsTitles=["Id","Predicted"]
submission=submission.reindex(columns=columnsTitles)
submission.to_csv('TFIDF_5W_Logistic.csv',index=0)
print(submission)



