from flask import render_template
from blog import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from flask import request
import pandas as pd
import numpy as np
import psycopg2
import pickle
from parse_comments_utils import text2features
from gensim import models, corpora


@app.route('/')
@app.route('/index')
def index():
    return render_template("post.html")

@app.route('/home')
def home():
    return render_template("post.html")

@app.route('/demo')
def demo():
    return render_template("demo.html")

@app.route('/blog_output')
def blog_output():
    X_input = request.args.get('input_comment')
    X_test = text2features(X_input)
    
    model_LR = pickle.load(open("/Users/mengyuan/Documents/Insight/InsightCherryPick/Processing/RF_tfidf50_num30depth10_0.527019540078.p", "rb"))
    predict_prob_LR = model_LR.predict_proba(X_test)
    predicted_LR = model_LR.predict(X_test)
    if predicted_LR == 1: #predict_prob_LR[0][1] >= 0.527019540078:
      purchase_intent = ["This comment shows:","purchase intent","."]
    else:
      purchase_intent = ["This comment shows:","NO purchase intent","."]

    model_kNN = pickle.load(open("/Users/mengyuan/Documents/Insight/InsightCherryPick/Processing/RFmulti_tfidf50_num150depth20.p", "rb"))
    predicted_kNN = model_kNN.predict(X_test)
    if predicted_kNN == -2:
      label = ["This is", "SPAM","."]
    elif predicted_kNN == -1:
      label = ["Customer shows", "NEGATIVITY", "about the product."]
    elif predicted_kNN == 0:
      label = ["Customer is", "AWARE of/NEUTRAL", "about the product."]
    elif predicted_kNN == 1:
      label = ["Customer is", "INTERESTED", "in the product."]
    elif predicted_kNN == 2:
      label = ["Customer is in", "CONSIDERATION/EVALUATION", "of the product."]
    elif predicted_kNN == 3:
      label = ["Customer shows", "PURCHASE INTENT", "of the product."]
    else:
      label = ["Customer has", "PURCHASED", "the product."]
    return render_template("post_output.html", purchase_probability = str(predict_prob_LR[0][1]), purchase_intent = purchase_intent, label = label, X_input = X_input)

