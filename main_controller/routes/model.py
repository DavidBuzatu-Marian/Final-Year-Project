from logging import debug, error
from flask import request
import sys
from flask.json import jsonify

from app import app
from app import mongo

try:
    from helpers.environment_helpers import *
    from helpers.request_helpers import *
    from helpers.cotroller_hepers import *
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))

# TODO: Change this to a parameter during training
MAX_TRIALS = 10
REQUIRED_INSTANCES = 2


@app.route("/model/train", methods=["POST"])
def model_train():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)
    available_instanecs = get_available_instances(
        environment, MAX_TRIALS, REQUIRED_INSTANCES
    )
