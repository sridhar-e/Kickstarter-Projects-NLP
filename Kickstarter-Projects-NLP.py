# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:23:16 2019

@author: Sridhar
"""

#Download Dataset - https://drive.google.com/open?id=1m95xx-zijHJEh4khG40wge01mP8tXbzm

import pandas as pd
import numpy as np

##import data - Set Work Directory First
dataset=pd.read_csv('NLP_Dataset.csv')

##check any missing value 
dataset.isna().sum()

##remove missing value
dataset=dataset.dropna()

##drop first column
dataset=dataset.drop(['Unnamed: 0'],axis=1)

##preprocessing - Conversion / Expansion of the Short form Definitions
import re
contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

##Creating a Function for the Created Expansion Code
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

#Importing NLTK
import nltk

#Creting Function for Normalization
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = doc.lower() #Converting the Data into Lowercase
    doc = expand_contractions(doc) #Apply the Created Contraction
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    
    doc = doc.strip()
    # tokenize document - Split the Sentence into Words
    tokens = nltk.word_tokenize(doc)
    
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

dataset['blurb'] = dataset['blurb'].apply(normalize_document)

# build train and test datasets
Text = dataset['blurb'].values
Label = dataset['state'].values

train_Text = Text[:150857] #70% of the Size of the Data (215510) i.e 150857
train_Label = Label[:150857] #70% of the Size of the Data (215510) i.e 150857

test_Text = Text[150857:] #Remaining 30% of the Size of the Data (215510) from 150857
test_Label = Label[150857:] #Remaining 30% of the Size of the Data (215510) from 150857

from sklearn.feature_extraction.text import TfidfVectorizer

# build TFIDF features on train reviews
tv = TfidfVectorizer(use_idf=True, min_df=5, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)
tv_train_features = tv.fit_transform(train_Text)

# transform test reviews into features
tv_test_features = tv.transform(test_Text)

print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)

# Logistic Regression model on TF-IDF features
from sklearn.linear_model import LogisticRegression

# instantiate model
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, solver='lbfgs', random_state=42)

# train model
lr.fit(tv_train_features, train_Label)

# predict on test data
lr_tfidf_predictions = lr.predict(tv_test_features)

labels = ['failed', 'successful']

##finding accuracy using confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(test_Label, lr_tfidf_predictions))
pd.DataFrame(confusion_matrix(test_Label, lr_tfidf_predictions), index=labels, columns=labels)

##accuracy
from sklearn.metrics import accuracy_score
accuracy_score(test_Label, lr_tfidf_predictions)

































