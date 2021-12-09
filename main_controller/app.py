from flask import Flask
import sys
import os
from dotenv import load_dotenv
from flask_pymongo import PyMongo

sys.path.insert(0, "./environment/")

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

import routes.environment


@app.route("/")
def hello_word():
    return "Hello from main controller"
