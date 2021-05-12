import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import gzip
import nltk
import re
import keras.backend as K
import pickle
from flask import Flask, request



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import MLkNN

from sklearn.feature_extraction.text import CountVectorizer
from RedditScrape import preprocess



with open('hobbies_classifier.obj', 'rb') as hf:
    hobbies_classifier = pickle.load(hf)

with open('kmeans_clusterer.obj', 'rb') as f:
    kmeans_clusterer = pickle.load(f)
 
with open('count_vectorizer.obj', 'rb') as cv_open:
    vectorizer = pickle.load(cv_open)


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
    
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
    
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def preprocess_personality_dataframe(posts):
    lemmatizer = WordNetLemmatizer()
    
    new_posts = cleanHtml(posts)
    new_posts = cleanPunc(new_posts)
    new_posts = keepAlpha(new_posts)
   
    
    return_posts = " ".join(lemmatizer.lemmatize(word, pos="v") for word in new_posts.split())
    return return_posts
    

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to our Tinder clone!"
    
@app.route("/newdict")
def predict():
    text = request.args.get('text')    
    text = preprocess(text)

    doc_term_matrix = vectorizer.transform([text])
    
    results = hobbies_classifier.predict_proba(doc_term_matrix)
    
    print("RESULTS: ", results)
    
    cluster = kmeans_clusterer.predict(results)
    
    return "Results: " + str(results) + " Cluster: " + str(cluster)
    
if __name__ == "__main__":
    app.run(debug=True)
   


