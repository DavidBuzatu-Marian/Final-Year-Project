from logging import error
from flask import Flask, request
from flask.json import jsonify
import os
import sys
from helpers.app_helpers import read_model_from_path, save_file
import torch

try:
    from nn_model import NNModel
except ImportError as exc:
    sys.stderr.write("Error: failed to import nnmodel module ({})".format(exc))

from app import app


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
