import os
from nn_loss.nn_loss_factory import NNLossFactory
from nn_optimizer.nn_optimizer_factory import NNOptimizerFactory

import torch
from flask.json import jsonify
from logging import error
from werkzeug.utils import secure_filename
from pathlib import Path


def read_model_from_path(path):
    if os.path.isfile(path):
        model = torch.load(path)
        return jsonify(repr(model._model))
    raise Exception("Error. No model found")


def save_file(file, path_env):
    filename = secure_filename(file.filename)
    path = os.getenv(path_env)
    Path(path).mkdir(parents=True, exist_ok=True)
    file.save(os.path.join(path, filename))


def get_loss(request_json):
    loss_factory = NNLossFactory()
    if not ("loss" in request_json):
        raise Exception("No loss function set")
    loss = request_json["loss"]
    return loss_factory.get_loss(
        loss_type=loss["loss_type"], parameters=loss["parameters"]
    )


def get_optimizer(request_json):
    optimizer_factory = NNOptimizerFactory()
    if not ("optimizer" in request_json):
        raise Exception("No optimizer set")
    optimizer = request_json["optimizer"]
    return optimizer_factory.get_optimizer(
        optimizer_type=optimizer["optimizer_type"], parameters=optimizer["parameters"]
    )


def get_hyperparameters(request_json):
    # TODO: Think how to represent hyperparams
    return {"epochs": 2, "num_workers": 1, "batch_size": 2, "shuffle": True}
