from logging import error
from flask import Flask, request
from flask.json import jsonify
import torch
import sys
import os
from dotenv import load_dotenv
from nn_helpers.nn_helpers import read_model_from_path
from flask_pymongo import PyMongo
from bson import json_util
import json

sys.path.insert(0, "./nn_model/")

try:
    from nn_model import NNModel
except ImportError as exc:
    sys.stderr.write("Error: failed to import nnmodel module ({})".format(exc))

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)


@app.route("/model/create", methods=["POST"])
def model_create():
    model = NNModel(request.json)
    path = os.getenv("MODEL_PATH")
    if os.path.isfile(path):
        os.remove(path)
    torch.save(model, path)
    return jsonify(repr(model._model))


@app.route("/model/train", methods=["POST"])
def model_train():
    path = os.getenv("MODEL_PATH")
    model = read_model_from_path(path)
    error(request.content_type)
    if not ("multipart/form-data" in request.content_type):
        return (
            jsonify("Content type is not supported. Please return multipart/form-data"),
            415,
        )
    error(request.files)
    return jsonify("tests")
    # Get data from database/local store
    # Get optimizer from request
    # Get loss from request
    # Get training options - epochs - currently set to default of 5
    # Save model


@app.route("/")
def hello_world():
    path = os.getenv("MODEL_PATH")
    return read_model_from_path(path)


# TODO: Create function for json return of database result: jsonify(json_util.dumps(([doc for doc in mongo.db.users.find()]))
