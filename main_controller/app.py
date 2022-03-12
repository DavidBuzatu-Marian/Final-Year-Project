from flask import Flask
import sys
import os
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import logging
import json

sys.path.insert(0, "./environment/")
sys.path.insert(1, "./helpers/")
sys.path.insert(2, "./nn_model/")
sys.path.insert(3, "./config/")

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

statuses = json.load(open("./config/statuses.json"))


logging.basicConfig(
    filename='./logs/general.log', level=logging.INFO,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.basicConfig(
    filename='./logs/error.log', level=logging.ERROR,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.basicConfig(filename='./logs/critical.log', level=logging.CRITICAL,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


import routes.environment
import routes.model
import routes.health
