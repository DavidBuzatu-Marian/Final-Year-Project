from logging import error
from flask import Flask, request
from flask.json import jsonify
import torch
import sys
import os
from dotenv import load_dotenv
from helpers.app_helpers import read_model_from_path

# from flask_pymongo import PyMongo
import json


sys.path.insert(0, "./nn_model/")


load_dotenv()

app = Flask(__name__)
# app.config["MONGO_URI"] = os.getenv("MONGO_URI")
# mongo = PyMongo(app)

import routes.model
import routes.dataset
import routes.instance
