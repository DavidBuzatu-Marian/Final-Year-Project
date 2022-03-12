from flask import Flask
import sys
import os
from dotenv import load_dotenv
from flask_pymongo import PyMongo
import logs.logger
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


import routes.environment
import routes.model
import routes.health
