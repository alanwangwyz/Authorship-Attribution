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
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train_data = None
test_data = None

### load data
def load_data():
    global train_data, test_data
    train_data = pd.read_csv('train_tweets.txt', delimiter="\t", header = None, quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv('test_tweets_unlabeled.txt', delimiter="\t", header = None, quoting=csv.QUOTE_NONE)
    
load_data()
train_data.columns = ['label','Tweet']
test_data.columns = ['Tweet']

def filter_fun(line):
    line = re.sub(r'[^a-zA-Z]',' ',line)
    line = line.lower()
    return line

train_data['Tweet'] = train_data['Tweet'].apply(filter_fun)
test_data['Tweet'] = test_data['Tweet'].apply(filter_fun)

train_data['Tweet'] = train_data['Tweet'].str.strip()  
test_data['Tweet'] = test_data['Tweet'].str.strip()

train_data['Tweet'] = train_data['Tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w) > 3]))
test_data['Tweet'] = test_data['Tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w) > 3]))

train_data['Tweet'] = train_data['Tweet'].str.split()
test_data['Tweet'] = test_data['Tweet'].str.split()
    
from nltk.stem.porter import *
stemmer = PorterStemmer()
train_data['Tweet'] = train_data['Tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
test_data['Tweet'] = test_data['Tweet'].apply(lambda x: [stemmer.stem(i) for i in x])

train_data['Tweet'] = train_data['Tweet'].apply(lambda x:" ".join(x))
test_data['Tweet'] = test_data['Tweet'].apply(lambda x:" ".join(x))


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 50000)
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

