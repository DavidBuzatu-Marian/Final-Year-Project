import os
import torch
from flask.json import jsonify
from logging import error


def get_params_from_list(parameters, parameters_set):
    return dict(
        filter(
            lambda key_value: (key_value[0] in parameters_set),
            parameters.items(),
        )
    )


def read_model_from_path(path):
    if os.path.isfile(path):
        model = torch.load(path)
        return jsonify(repr(model._model))
    raise error("Error. No model found")
