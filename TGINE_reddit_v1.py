# -*- coding: utf-8 -*-
"""
Raquel 
Tecnologías de Gestión de la información no estructurada
USC - Master Big Data
"""

import praw
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

reddit = praw.Reddit(client_id=$client_id,
                     client_secret=$client_secret,
                     user_agent=$user_agent)
subreddit = reddit.subreddit('AskScience')

def parse_submissions(submissions_list):
    list_of_posts=[]
    for submission in submissions_list:
        post = {}
        list_of_comments = []
        post['date']=submission.created_utc
        try:
            post['author']=submission.author.name
        except AttributeError:
                continue  
        post['title']=submission.title
        post['body']=submission.selftext
        submission.comments.replace_more(limit=None)
        for top_level_comment in submission.comments:
            comment = {}
            comment['date']=top_level_comment.created_utc
            try:
                comment['author']=top_level_comment.author.name
            except AttributeError:
                comment['author']='Deleted'
            comment['body']=top_level_comment.body
            list_of_comments.append(comment)
        post['comments']=list_of_comments
        list_of_posts.append(post)
    return list_of_posts

def write_to_file(filename, list_to_write):
    with open(filename, 'w') as outfile:
        json.dump(list_to_write, outfile)

def read_from_file(filename):
    with open(filename) as infile:
        return json.load(infile)
    
def create_corpus(a_read_list_of_posts):
    list_of_documents=[]
    for element in a_read_list_of_posts:
        list_of_comments=[]
        for comment in element['comments']:
            list_of_comments.append(comment['body'])
        comments=' '.join(list_of_comments)
        list_of_documents.append(' '.join([element['title'], element['body'],comments]))
    return list_of_documents

def vectorize_analyze(corpus):
    print("\nNúmero de documentos:\n{}".format(len(corpus)))
    tf = TfidfVectorizer(stop_words='english', min_df=10)
    try:
        fitted_corpus=tf.fit_transform(corpus)
    except ValueError:
        print("El número de posts es insuficiente para el análisis")
        return
    print("\nNúmero de palabras total:\n{}".format(len(tf.vocabulary_)))
    feature_names = np.array(tf.get_feature_names())
    max_values = fitted_corpus.max(axis=0).toarray().ravel() 
    sort_by_tfidf = max_values.argsort()
    print("\n10 palabras con tfidf más bajo:\n{}".format(feature_names[sort_by_tfidf[:10]]))
    print("\n50 palabras con tfidf más alto: \n{}".format(feature_names[sort_by_tfidf[-50:]]))
    tf100 = TfidfVectorizer(stop_words='english', min_df=10, max_features=100)
    tf100.fit_transform(corpus)
    print("\n100 términos más repetidos:\n {}".format(tf100.get_feature_names()))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", help="Número de posts a extraer", type=int)
    args = parser.parse_args()
    limit=args.limit #Límite de posts a extraer, con None se extrae el máximo
    try:
        read_list_of_posts_new=read_from_file('askscience_new.json')
    except FileNotFoundError:
        print("No se ha encontrado fichero para new, se extraen posts de Reddit")
        list_of_posts_new=parse_submissions(subreddit.new(limit=limit))
        write_to_file('askscience_new.json', list_of_posts_new)
        read_list_of_posts_new=read_from_file('askscience_new.json')
        
    try:
        read_list_of_posts_top=read_from_file('askscience_top.json')
    except FileNotFoundError:
        print("No se ha encontrado fichero para top, se extraen posts de Reddit")
        list_of_posts_top=parse_submissions(subreddit.top(limit=limit))
        write_to_file('askscience_top.json', list_of_posts_top)
        read_list_of_posts_top=read_from_file('askscience_top.json')
        
    try:
        read_list_of_posts_rising=read_from_file('askscience_rising.json')
    except FileNotFoundError:
        print("No se ha encontrado fichero para rising, se extraen posts de Reddit")
        list_of_posts_rising=parse_submissions(subreddit.rising(limit=limit))
        write_to_file('askscience_rising.json', list_of_posts_rising)
        read_list_of_posts_rising=read_from_file('askscience_rising.json')
          
    try:
        read_list_of_posts_hot=read_from_file('askscience_hot.json')
    except FileNotFoundError:
        print("No se ha encontrado fichero para hot, se extraen posts de Reddit")
        list_of_posts_hot=parse_submissions(subreddit.hot(limit=limit))
        write_to_file('askscience_hot.json', list_of_posts_hot)
        read_list_of_posts_hot=read_from_file('askscience_hot.json')
               
        
    # Se crean los corpus para las tres extracciones
    corpus_new=create_corpus(read_list_of_posts_new)
    corpus_top=create_corpus(read_list_of_posts_top)
    corpus_rising=create_corpus(read_list_of_posts_rising)
    corpus_hot=create_corpus(read_list_of_posts_hot)
    
    #Se vectorizan y analizan
    vectorize_analyze(corpus_new) 
    vectorize_analyze(corpus_hot)
    vectorize_analyze(corpus_rising)
    vectorize_analyze(corpus_top)
    