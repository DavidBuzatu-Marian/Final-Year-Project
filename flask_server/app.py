from flask import Flask
from flask import request
from flask.helpers import url_for
from flask.json import jsonify
import redis
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, "./nn_model")

from nn_model import NNModel

load_dotenv()

redis_instance = redis.Redis(
    host=os.environ.get("CACHE_REDIS_HOST"),
    port=os.environ.get("CACHE_REDIS_PORT"),
    db=os.environ.get("CACHE_REDIS_DB"),
)
app = Flask(__name__)


@app.route("/model/create", methods=["POST"])
def model_create():
    model = NNModel(request.json)
    redis_instance.set("model", model)
    # TODO: Check if there is a saved model and delete it
    return jsonify(model)


@app.route("/")
def hello_world():
    return "Hello!"
