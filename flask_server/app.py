from logging import error
from flask import Flask, request
from flask.json import jsonify
import torch
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, "./nn_model/")

try:
    from nn_model import NNModel
except ImportError as exc:
    sys.stderr.write("Error: failed to import nnmodel module ({})".format(exc))

load_dotenv()

app = Flask(__name__)


@app.route("/model/create", methods=["POST"])
def model_create():
    model = NNModel(request.json)
    # TODO: Check if there is a saved model and delete it
    path = os.getenv("MODEL_PATH")
    if os.path.isfile(path):
        os.remove(path)
    torch.save(model, path)
    return jsonify(repr(model._model))


@app.route("/")
def hello_world():
    path = os.getenv("MODEL_PATH")
    if os.path.isfile(path):
        model = torch.load(path)
        return jsonify(repr(model._model))
    raise error("Error. No model found")
