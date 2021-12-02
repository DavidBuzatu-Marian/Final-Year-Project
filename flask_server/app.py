from logging import error
from flask import Flask, request
from flask.json import jsonify
import torch
import sys
import os
from dotenv import load_dotenv
from helpers.app_helpers import read_model_from_path
from flask_pymongo import PyMongo
import json
from werkzeug.utils import secure_filename

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
    # Get data from database/local store
    # Get optimizer from request
    # Get loss from request
    # Get training options - epochs - currently set to default of 5
    # Save model


@app.route("/dataset/add", methods=["POST"])
def dataset_add():
    if not ("multipart/form-data" in request.content_type):
        return (
            jsonify("Content type is not supported. Please return multipart/form-data"),
            415,
        )
    if len(request.files) == 0:
        return (jsonify("At least a file needs to be selected"), 400)
    for file in request.files.getlist("train"):
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.getenv("TRAIN_DATA_PATH"), filename))
    for file in request.files.getlist("validation"):
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.getenv("TRAIN_DATA_PATH"), filename))
    return jsonify("tests")


@app.route("/")
def hello_world():
    path = os.getenv("MODEL_PATH")
    return read_model_from_path(path)


# TODO: Create function for json return of database result: jsonify(json_util.dumps(([doc for doc in mongo.db.users.find()]))
