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
    from helpers.controller_hepers import *
    from environment_classes.target_environment import TargetEnvironment
    from routes.error_handlers.server_errors_handler import return_500_on_uncaught_server_error
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


@app.route("/model/train", methods=["POST"])
@return_500_on_uncaught_server_error
def model_train():
    update_environment_status(mongo.db, target_environment, "3")
    target_environment = TargetEnvironment(
        get_user_id(request.json),
        get_environment_id(request.json))
    environment = get_environment(mongo.db, target_environment)
    training_options = get_training_options(request.json)
    available_instances = get_available_instances(
        environment, training_options['max_trials'], training_options['required_instances']
    )

    training_iterations = get_training_iterations(request.json)
    instance_training_parameters = get_instance_training_parameters(request.json)
    return train_model(
        mongo.db,
        random.sample(list(available_instances), training_options['required_instances']),
        training_iterations,
        instance_training_parameters,
        target_environment
    )


@app.route("/model/create", methods=["POST"])
@return_500_on_uncaught_server_error
def model_create():
    target_environment = TargetEnvironment(
        get_user_id(request.json),
        get_environment_id(request.json))
    environment = get_environment(mongo.db, target_environment)
    model_network_options = get_model_network_options(request.json)
    create_model(environment["environment_ips"], model_network_options)

    return "Created models on instances"


@app.route("/model/loss", methods=["POST"])
@return_500_on_uncaught_server_error
def model_loss():
    target_environment = TargetEnvironment(
        get_user_id(request.json),
        get_environment_id(request.json))
    environment = get_environment(mongo.db, target_environment)
    instance_training_parameters = get_instance_training_parameters(request.json)
    losses = compute_losses(
        instance_training_parameters, environment["environment_ips"]
    )
    return json.dumps(losses)
