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
        return model
    raise Exception("Error. No model found")


def save_file(file, path_env):
    filename = secure_filename(file.filename)
    path = os.getenv(path_env)
    if not Path(path).is_dir():
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


def get_optimizer(request_json, model=None):
    optimizer_factory = NNOptimizerFactory()
    if not ("optimizer" in request_json):
        raise Exception("No optimizer set")
    optimizer = request_json["optimizer"]
    optimizer["parameters"]["params"] = model
    return optimizer_factory.get_optimizer(
        optimizer_type=optimizer["optimizer_type"], parameters=optimizer["parameters"]
    )


def get_hyperparameters(request_json):
    request_json["hyperparameters"].setdefault("epochs", 10)
    request_json["hyperparameters"].setdefault("num_workers", 1)
    request_json["hyperparameters"].setdefault("batch_size", 1)
    request_json["hyperparameters"].setdefault("shuffle", True)
    return request_json["hyperparameters"]

def get_probability_of_failure(request_json):
    return request_json["probability_of_failure"]

def get_processors(request_json):
    # TODO: Find a way to encapsulate this
    return [torch.argmax]


def process_output(output, processors):
    for proc in processors:
        output = proc(output, dim=1)
    return output


def get_loss_type(request_json):
    if request_json["loss_type"] == "training":
        return {"data_path": "TRAIN_DATA_PATH", "labels_path": "TRAIN_LABELS_PATH"}
    if request_json["loss_type"] == "test":
        return {"data_path": "TEST_DATA_PATH", "labels_path": "TEST_LABELS_PATH"}
    return {
        "data_path": "VALIDATION_DATA_PATH",
        "labels_path": "VALIDATION_LABELS_PATH",
    }
