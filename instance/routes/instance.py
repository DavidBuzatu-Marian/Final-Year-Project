from logging import error
from flask import Flask, request
from flask.helpers import send_file
from flask.json import jsonify
import os
import sys

import torch
import yaml

try:
    from nn_model_factory.nn_model import NNModel
    from helpers.data_helpers import *
    from helpers.app_helpers import *
    from routes.error_handlers.server_errors_handler import return_500_on_uncaught_server_error
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))

from app import app


@app.route("/instance/availability", methods=["GET"])
@return_500_on_uncaught_server_error
def instance_availability():
    # For now, always available
    return jsonify({"availability": True})


@app.route('/instance/probabilityoffailure', methods=["POST"])
@return_500_on_uncaught_server_error
def set_probability_of_failure():
    env_configuration = {"probabilityOfFailure": 0.0}
    with open(os.getenv("INSTANCE_CONFIG_FILE_PATH"), "r") as yaml_config_file:
        env_configuration = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
        env_configuration["probabilityOfFailure"] = get_probability_of_failure(request.json)
    with open(os.getenv("INSTANCE_CONFIG_FILE_PATH"), "w") as yaml_config_file:
        yaml.dump(env_configuration, yaml_config_file)
    return "Saved probability of failure"
