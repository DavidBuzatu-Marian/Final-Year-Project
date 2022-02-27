from logging import error
from flask import request
import sys
from flask.json import jsonify
import random
from app import app
from app import mongo

try:
    from helpers.environment_helpers import *
    from helpers.request_helpers import *
    from helpers.cotroller_hepers import *
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))

@app.route("/model/train", methods=["POST"])
def model_train():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    environment = get_environment(mongo.db, environment_id, user_id)
    training_options = get_training_options(request.json)
    available_instances = get_available_instances(
        environment, training_options.max_trials, training_options.required_instances
    )
    training_iterations = get_training_iterations(request.json)
    instance_training_parameters = get_instance_training_parameters(request.json)
    return train_model(
        random.sample(available_instances, training_options.required_instances),
        training_iterations,
        instance_training_parameters,
    )


@app.route("/model/create", methods=["POST"])
def model_create():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    environment = get_environment(mongo.db, environment_id, user_id)
    model_network_options = get_model_network_options(request.json)
    create_model(environment["environment_ips"], model_network_options)

    return "Created models on instances"


@app.route("/model/loss", methods=["POST"])
def model_loss():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    environment = get_environment(mongo.db, environment_id, user_id)
    instance_training_parameters = get_instance_training_parameters(request.json)
    losses = compute_losses(
        instance_training_parameters, environment["environment_ips"]
    )
    return json.dumps(losses)
