from logging import error
from flask import Flask, request
from flask.helpers import send_file
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


@app.route("/instance/availability", methods=["GET"])
def instance_availability():
    # For now, always available
    return jsonify({"availability": True})


@app.route('/instance/probabilityoffailure', methods=["POST"])
def set_probability_of_failure():
    os.environ['PROBABILITY_OF_FAILURE'] = get_probability_of_failure(request.json)
    return "Saved probability of failure"