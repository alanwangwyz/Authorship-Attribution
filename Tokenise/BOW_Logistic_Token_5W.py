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


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_features = 50000, stop_words = 'english', encoding = 'utf-8') 
X_train = bow_vectorizer.fit_transform(train_data['Tweet'])
X_test = bow_vectorizer.fit_transform(test_data['Tweet'])

print (bow_vectorizer.get_feature_names())

#######
lreg = LogisticRegression()
lreg.fit(X_train, train_data['label']) # training the model
 
trainPrediction = lreg.predict_proba(X_test) # predicting on the validation set

prediction = lreg.predict(X_test)
print(prediction)

#######
test_data['Predicted'] = prediction
submission = test_data[['Predicted']]
submission.index = np.arange(1, len(submission) + 1)
submission['Id'] = submission.index

columnsTitles=["Id","Predicted"]
submission=submission.reindex(columns=columnsTitles)
submission.to_csv('BOW_Logistic_Token_5W.csv',index=0)
print(submission)

#######
def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()
    
save_pickle(lreg, os.path.join('BOW_Logistic_Token_5W.p'))
