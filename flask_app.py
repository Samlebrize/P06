import datetime
import calendar
import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import flask


import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('wordnet')

rep = 'MultilabelBina.pkl'
with open(rep, 'rb') as file:
    multilabel_binarizer= pickle.load(file)

rep = 'Vectorizer.pkl'
with open(rep, 'rb') as file:
    vectorizer_X= pickle.load(file)



def avg_jacard(y_true,y_pred):
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    return jacard.mean()*100

def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Jacard score: {}".format(avg_jacard(y_test, y_pred)))
    print("---")   


rep = 'Best_Classifier.pkl'
with open(rep, 'rb') as file:
    clf= pickle.load(file)

rep = 'stopwords.pkl'
with open(rep, 'rb') as file:
    stop_words= pickle.load(file)


def formating_the_doc(doc):
  from gensim.utils import simple_preprocess
  doc = BeautifulSoup(doc).find_all('p')
  doc = [phrase.text for phrase in doc]
  doc = " ".join(doc)
  # Lowering
  doc = doc.lower()
  # Tokenizing
  tokenizer = RegexpTokenizer(r'\w+')
  doc = tokenizer.tokenize(doc)
  # Removing stop words and most frequent ones
  doc = [word for word in simple_preprocess(str(doc)) if word not in stop_words]
  # Lemmatization
  lemmatizer = WordNetLemmatizer()
  doc = [lemmatizer.lemmatize(token) for token in doc]
  return doc

def get_tags_supervised(Title,Question):
  Bow = Title + Question
  X_test1 = " ".join(Bow)
  X_test2 = [X_test1]
  X_test3 = vectorizer_X.transform(X_test2)
  y_pred = clf.predict(X_test3)
  Ltags = multilabel_binarizer.inverse_transform(y_pred)
  return Ltags

Tit = ['Cross-platform C# application using Mono with nice UI']
Ques = ["<p>I want to create a cross-platform C# application using Mono.</p>\n\n<p>I used to think that for these purposes I need to use GTK#, Now it seems to me that it is not as attractive to users as I had thought before.\nA good choice would be a Silverlight (Moonlight), but it's\nsolution for WEB and despite the fact that it can be run outside of the browser, I think that this technology not completely suitable for my purposes. I think WPF would be a great choice,\nbut Mono does not support it.</p>\n\n<p>So I'm looking for cross-platform gui toolkit for C# to build applications with powerful, animated and custom use"]


print(get_tags_supervised(Tit,Ques))

# Fonction flask
app = flask.Flask(__name__, template_folder='templates')
app.config["DEBUG"] = False


# Page d'accueuil (index), la fonction post lance le calcul de distance et affiche la page résultat

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        Title = str(flask.request.form['titl'])
        Question = str(flask.request.form['Quest'])
        Resultat = str(get_tags_supervised([Title],[Question]))
        return flask.render_template('resultats.html',Tags=Resultat)



# lancement de la fonction

if __name__ == '__main__':
    app.run()