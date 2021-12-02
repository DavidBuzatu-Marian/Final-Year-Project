import os
import torch
from flask.json import jsonify
from logging import error
from werkzeug.utils import secure_filename
from pathlib import Path


def read_model_from_path(path):
    if os.path.isfile(path):
        model = torch.load(path)
        return jsonify(repr(model._model))
    raise error("Error. No model found")


def save_file(file, path_env):
    filename = secure_filename(file.filename)
    path = os.getenv(path_env)
    Path(path).mkdir(parents=True, exist_ok=True)
    file.save(os.path.join(path, filename))
