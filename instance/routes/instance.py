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
    with open(os.getenv("INSTANCE_CONFIG_FILE_PATH"), "w+") as yaml_config_file:
        yaml.dump(get_probability_of_failure(request.json), yaml_config_file)
    return "Saved probability of failure"
