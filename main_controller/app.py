from flask import Flask
import sys
import os
from dotenv import load_dotenv
from flask_pymongo import PyMongo

sys.path.insert(0, "./environment/")
sys.path.insert(1, "./helpers/")
sys.path.insert(2, "./nn_model/")

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

import routes.environment
import routes.model
