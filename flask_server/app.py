from flask import Flask
from flask import request

import sys

sys.path.insert(0, "./nn_model")

from nn_model import NNModel

app = Flask(__name__)


@app.route("/model/create", methods=["POST"])
def model_create():
    model = NNModel(request.json)
    return repr(model._model)


@app.route("/")
def hello_world():
    return "Hello!"
