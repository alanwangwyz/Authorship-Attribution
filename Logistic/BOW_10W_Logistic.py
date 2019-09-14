import re, pickle, os, string
import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression

train_data = None
test_data = None


def load_data():
    global train_data, test_data
    train_data = pd.read_csv('train_tweets.txt', delimiter="\t", header=None, quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv('test_tweets_unlabeled.txt', delimiter="\t", header=None, quoting=csv.QUOTE_NONE)


load_data()
train_data.columns = ['label', 'Tweet']
test_data.columns = ['Tweet']

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_features=100000)

allData = train_data['Tweet'].values.tolist() + test_data['Tweet'].values.tolist()
bowAllData = bow_vectorizer.fit_transform(allData)
train_tagged = bow_vectorizer.get_feature_names()

lentrain = len(train_data)

train = bowAllData[:lentrain]
test = bowAllData[lentrain:]


lreg = LogisticRegression()
lreg.fit(train, train_data['label'])  # training the model

trainPrediction = lreg.predict_proba(test)  # predicting on the validation set

prediction = lreg.predict(test)

def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents, protocol = 4)
    save_documents.close()


test_data['Predicted'] = prediction
submission = test_data[['Predicted']]
submission.index = np.arange(1, len(submission) + 1)
submission = pd.read_csv('ResultBOW20W.csv', names = ['Id', 'Predicted'])
submission.to_csv('ResultBOW20W.csv', index=0)
save_pickle(lreg, os.path.join('20WBOW.p'))