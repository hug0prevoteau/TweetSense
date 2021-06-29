from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import json
from pymongo import MongoClient
import pymongo
import subprocess
from bson import json_util
import re
import sys

MONGO_HOST = "xxxx"

client = MongoClient(MONGO_HOST)
db = client.twitterdb
collection = db.twitter_search

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/sentiments/tweets/all')
def getAll():
    collection = db.twitter_search
    data = []
    for search in collection.find({"Prediction":{"$gt":0, "$lt": 1}},{"text":1, "Prediction":1, "_id":0, "created_at": 1}):
        data.append(search)
    return jsonify(data)

@app.route('/sentiments/tweets')
def tweets():
    collection = db.twitter_search
    data = []
    for search in collection.find({'text': { '$regex': re.compile(r"\b" + request.args.get('q') + r"\b", re.IGNORECASE)}, "Prediction":{"$gt":0, "$lt": 1}}, {"text":1, "Prediction":1, "_id":0, "created_at": 1}):
        data.append(search)
    return jsonify(data)

@app.route('/sentiments/articles')
def articles():
    collection = db.google_search
    data = []
    for search in collection.find({'text': { '$regex': re.compile(r"\b" + request.args.get('q') + r"\b", re.IGNORECASE)}, "Prediction":{"$gt":0, "$lt": 1}}, {"text":1, "Prediction":1, "_id":0, "created_at": 1}):
        data.append(search)
    return jsonify(data)

@app.route('/sentiments/tweets/last')
def last50():
    collection = db.twitter_search
    data = []
    for search in collection.find({"Prediction":{"$gt":0, "$lt": 1}}, {"text":1, "Prediction":1, "_id":0, "created_at": 1}).sort([("_id", -1)]).limit(int(request.args.get('max'))):
        data.append(search)
    return jsonify(json.loads(json_util.dumps(data)))
    
@app.route('/scripts', methods = ['POST'])
def scrap():
    if request.form.get('script') == 'ScrapTwitter':
        subprocess.Popen([sys.executable, "nlp_sentiment_analysis.py"])
        return jsonify("Scrapping en cours")
    if request.form.get('script') == 'Analyse':
        subprocess.Popen([sys.executable, "notebook_inference_bert.py"])
        return jsonify("Analyse en cours")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=50000)