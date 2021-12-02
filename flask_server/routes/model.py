from logging import error
from flask import Flask, request
from flask.json import jsonify
import os
import sys

import torch


try:
    from nn_model import NNModel
    from helpers.data_helpers import *
    from helpers.app_helpers import *
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))

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
    loss_func = get_loss(request.json)
    optimizer = get_optimizer(request.json)
    hyperparameters = get_hyperparameters(request.json)

    train_dataloader = get_dataloader(
        data_path=os.getenv("TRAIN_DATA_PATH"),
        labels_path=os.getenv("TRAIN_LABELS_PATH"),
        hyperparameters=hyperparameters,
    )
    for _ in range(0, hyperparameters["epochs"]):
        for data, label in train_dataloader:
            optimizer.zero_grad()

            data = reshape_data(data, hyperparameters)
            data = normalize_data(data, hyperparameters)

            output = model(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

    torch.save(model, path)
