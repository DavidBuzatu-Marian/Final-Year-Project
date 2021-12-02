import os
import torch
from flask.json import jsonify
from logging import error


def read_model_from_path(path):
    if os.path.isfile(path):
        model = torch.load(path)
        return jsonify(repr(model._model))
    raise error("Error. No model found")
