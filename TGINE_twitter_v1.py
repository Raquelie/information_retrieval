#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author Raquel
Tecnologías de Gestión de la información no estructurada
USC - Master Big Data
"""

import tweepy
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Twitter access token and users
ACCESS_TOKEN = $ACCESS_TOKEN
ACCESS_SECRET = $ACCESS_SECRET
CONSUMER_KEY = $CONSUMER_KEY
CONSUMER_SECRET = $CONSUMER_SECRET

users=["neiltyson", "ProfBrianCox", "joerogan", "StephenKing"]

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

#Functions for extracting data
def save_public_tweets(user, filename):
    tweet_list=[]
    for status in tweepy.Cursor(api.user_timeline, user).items():
        tweet_list.append(status._json)
    with open(filename, 'w') as outfile:
         json.dump(tweet_list, outfile)

def read_tweets_from_file(filename):
    with open(filename) as infile:
        return json.load(infile)

def extract_text_from_tweets(tweets):
    list_of_texts=[]
    for elem in tweets:
        list_of_texts.append(elem['text'])
    return list_of_texts

def preprocess_tweets(tweets, more_tweets):
    # test and train are exchanged in the scikit function so that the latest tweets correspond to the test data
    x_test_tmp, x_train_tmp=train_test_split(tweets, test_size=0.7, shuffle=False, stratify = None)
    x_test, x_train=train_test_split(more_tweets, test_size=0.7, shuffle=False, stratify = None)
    x_test=x_test+x_test_tmp
    x_train=x_train+x_train_tmp
    y_test=[0] * len(x_test_tmp)+[1]*(len(x_test)-len(x_test_tmp))
    y_train=[0] * len(x_train_tmp)+[1]*(len(x_train)-len(x_train_tmp))
    # returns data and labels for train and test
    return x_train, y_train, x_test, y_test

filenames=['user1.json', 'user2.json', 'user3.json', 'user4.json']
for user, filename in zip(users, filenames):
    save_public_tweets(user,filename)    
    
#extract text from json
dataset1=extract_text_from_tweets(read_tweets_from_file('user1.json'))
dataset2=extract_text_from_tweets(read_tweets_from_file('user2.json'))
dataset3=extract_text_from_tweets(read_tweets_from_file('user3.json'))
dataset4=extract_text_from_tweets(read_tweets_from_file('user4.json'))
print "Lengths: ", len(dataset1), len(dataset2), len(dataset3), len(dataset4)

# Sim for the first classification (similar users) and Dif for the second (different users)
xSim_train, ySim_train, xSim_test, ySim_test=preprocess_tweets(dataset1, dataset2)
xDif_train, yDif_train, xDif_test, yDif_test=preprocess_tweets(dataset3, dataset4)
print "Train lenghts:", len(xSim_train), len(xDif_train)
print "Test lengths: ", len(xSim_test), len(xDif_test)

# Feature extraction
vecSim=TfidfVectorizer(stop_words='english')
vecDif=TfidfVectorizer(stop_words='english')
xSim_counts_train = vecSim.fit_transform(xSim_train)
xDif_counts_train = vecDif.fit_transform(xDif_train)
# using the fit from training
xSim_counts_test = vecSim.transform(xSim_test)
xDif_counts_test = vecDif.transform(xDif_test)

#SVM classification
def classify_and_predict(C, kernel, xTrain, yTrain, xTest, yTest):
    clf= SVC(C=C, kernel=kernel)
    clf.fit(xTrain, yTrain) 
    pred=clf.predict(xTest)
    acc=accuracy_score(yTest, pred)
    cm=confusion_matrix(yTest, pred)
    return acc, cm

def search_classifier_parameters (ker, param, xTrain, yTrain, xTest, yTest):
    kernel=[]
    C=[]
    accuracy=[]
    confMatrix=[]
    for k in ker:
        for c in param:
            acc, cm=classify_and_predict(c, k, xTrain, yTrain, xTest, yTest)
            kernel.append(k)
            C.append(c)
            accuracy.append(acc)
            confMatrix.append(cm)
    df=pd.DataFrame({'kernel': kernel,
         'C': C,
         'accuracy': accuracy,
         'confusion_matrix': confMatrix 
        })
    print df
    print "\nBest parameters:"
    print df.loc[df['accuracy'].idxmax()]
    print "\nConfusion matrix\n",  df.loc[df['accuracy'].idxmax()]['confusion_matrix']
    return df.loc[df['accuracy'].idxmax()]

#Parameters to iterate SVC
ker=['linear', 'poly', 'rbf']
param=[0.001, 0.01, 0.1, 1, 10, 100, 1000]

#Dataset composed of two people with similar profile
print "\nResults for similar set in training:\n"
best_param=search_classifier_parameters(ker, param, xSim_counts_train, ySim_train, xSim_counts_train, ySim_train)

print "\nBest accuracy and confusion matrix for test data in similar set"
acc, m=classify_and_predict(best_param['C'], best_param['kernel'], xSim_counts_train, ySim_train, xSim_counts_test, ySim_test)
print acc
print m

#Dataset composed of two people with different profile
print "\nResults for different set in training:\n"
best_param=search_classifier_parameters(ker, param, xDif_counts_train, yDif_train, xDif_counts_train, yDif_train)

print "\nBest accuracy and confusion matrix for test data in different set"
acc, m=classify_and_predict(best_param['C'], best_param['kernel'], xDif_counts_train, yDif_train, xDif_counts_test, yDif_test)
print acc
print m

# Sentiment analysis
pos_words=[]
with open('positive-words.txt','r') as f:
     for curline in f:
         if curline.startswith(';'):
            continue
         elif curline.startswith('\r'):
            continue
         else:
            try:
                curline.decode('UTF-8', 'strict')
                pos_words.append(curline.rstrip('\r\n'))
            except UnicodeDecodeError:
                print "Line skipped:", curline  
        
neg_words=[]
with open('negative-words.txt','r') as f:
     for curline in f:
         if curline.startswith(';'):
            continue
         elif curline.startswith('\r'):
            continue
         else:
            try:
                curline.decode('UTF-8', 'strict')
                neg_words.append(curline.rstrip('\r\n'))
            except UnicodeDecodeError:
                print "Line skipped:", curline       
                
vecPos=CountVectorizer()
vecPos.fit(pos_words)
vecNeg=CountVectorizer()
vecNeg.fit(neg_words)

def sentiment_analysis (vecPos, vecNeg, data):
    p=vecPos.transform(data)

    vecTotal=CountVectorizer()
    t=vecTotal.fit_transform(data)
    print "Porcentaje de palabras positivas: ", float(p.sum())/float(t.sum())*100

    
    n=vecNeg.transform(data)
    print "Porcentaje de palabras negativas: ", float(n.sum())/float(t.sum())*100

print "Dataset 1"
sentiment_analysis(vecPos, vecNeg, dataset1)
print "\nDataset 2"
sentiment_analysis(vecPos, vecNeg, dataset2)
print "\nDataset 3"
sentiment_analysis(vecPos, vecNeg, dataset3)
print "\nDataset 4"
sentiment_analysis(vecPos, vecNeg, dataset4)