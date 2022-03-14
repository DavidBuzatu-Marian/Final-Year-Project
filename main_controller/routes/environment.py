from logging import debug, error
from re import T
from flask import request
import sys
from flask.json import jsonify
import random
import json

from app import app
from app import mongo


try:
    from environment_classes.environment import Environment
    from environment_classes.target_environment import TargetEnvironment
    from helpers.environment_helpers import *
    from helpers.request_helpers import *
    from routes.error_handlers.server_errors_handler import return_500_environment_critical_error, return_500_on_uncaught_server_error
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


@app.route("/environment/create", methods=["POST"])
@return_500_environment_critical_error
def environment_create():
    environment = Environment(request.json)
    user_id = get_user_id(request.json)
    environment_id = save_environment_for_user(mongo.db, user_id, environment)
    target_environment = TargetEnvironment(user_id, environment_id)
    add_environment_id_to_request(environment_id)
    apply_terraform(user_id, environment)
    output = get_terraform_output()
    json_output = to_json(output)
    send_options_to_instances(
        json_output["gci_instances_ids"]["value"],
        environment.environment_options)
    save_ips_for_user(
        mongo.db, json_output["gci_instances_ids"], target_environment
    )
    create_environment_data_distribution_entry(
        mongo.db,
        json_output["gci_instances_ids"],
        target_environment
    )
    return "Created {} environments with requested options. Environments are ready for receiving datasets".format(
        environment.nr_instances
    )


@app.route("/environment/delete", methods=["DELETE"])
@return_500_environment_critical_error
def environment_delete():
    target_environment = TargetEnvironment(
        get_user_id(request.json),
        get_environment_id(request.json))
    update_environment_status(mongo.db, target_environment, "6")
    delete_environment(mongo.db, target_environment)
    return "Destroyed environment {} for user {}".format(
        target_environment.id, target_environment.user_id)


@app.route("/environment/dataset/data", methods=["POST"])
@return_500_on_uncaught_server_error
def environment_dataset_data():
    target_environment = TargetEnvironment(
        get_user_id(request.args),
        get_environment_id(request.args))
    environment = get_environment(mongo.db, target_environment)
    update_environment_status(mongo.db, target_environment, "2")
    environment_data_distribution = get_environment_data_distribution(
        mongo.db, target_environment
    )
    for environment_ip, _ in environment_data_distribution["distributions"].items():
        if not (environment_ip in environment["environment_ips"]):
            return (jsonify("Environment ip is invalid"), 400)
    instances_data = post_data_distribution(
        request.files, environment_data_distribution["distributions"]
    )
    save_environment_data_distribution(
        mongo.db, target_environment, instances_data
    )
    update_environment_status(mongo.db, target_environment, "5")
    return "Saved data in instances"


@app.route("/environment/dataset/validation", methods=["POST"])
@return_500_on_uncaught_server_error
def environment_dataset_validation():
    target_environment = TargetEnvironment(
        get_user_id(request.args),
        get_environment_id(request.args))
    environment = get_environment(mongo.db, target_environment)
    update_environment_status(mongo.db, target_environment, "2")
    instances_data = post_data_to_instance(
        request.files, environment["environment_ips"]
    )
    save_environment_data_distribution(
        mongo.db, target_environment, instances_data
    )
    update_environment_status(mongo.db, target_environment, "5")
    return "Saved validation data in instances"


@app.route("/environment/dataset/test", methods=["POST"])
@return_500_on_uncaught_server_error
def environment_dataset_test():
    target_environment = TargetEnvironment(
        get_user_id(request.args),
        get_environment_id(request.args))
    environment = get_environment(mongo.db, target_environment)
    update_environment_status(mongo.db, target_environment, "2")
    instances_data = post_data_to_instance(
        request.files, environment["environment_ips"]
    )
    save_environment_data_distribution(
        mongo.db, target_environment, instances_data
    )
    update_environment_status(mongo.db, target_environment, "5")
    return "Saved test data in instances"


@app.route("/environment/dataset/distribution", methods=["POST"])
@return_500_on_uncaught_server_error
def environment_dataset_distribution():
    target_environment = TargetEnvironment(
        get_user_id(request.json),
        get_environment_id(request.json))
    environment = get_environment(mongo.db, target_environment)
    environment_data_distribution = get_data_distribution(request.json)
    dataset_length = get_dataset_length(request.json)

    # TODO: Choose between random distribution or given distribution
    for environment_ip, distribution in environment_data_distribution.items():
        if not (environment_ip in environment["environment_ips"]):
            abort_with_text_response(400, "Environment IP is invalid")
        environment_data_distribution[environment_ip] = random.sample(
            range(0, dataset_length), distribution
        )
    save_environment_test_data_distribution(
        mongo.db, target_environment, environment_data_distribution
    )
    return "Saved dataset distribution"
