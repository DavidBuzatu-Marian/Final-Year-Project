from logging import debug, error
from flask import request
import sys
from flask.json import jsonify
import random
import json

from app import app
from app import mongo


try:
    from environment import Environment
    from helpers.environment_helpers import *
    from helpers.request_helpers import *
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


@app.route("/environment/create", methods=["POST"])
def environment_create():
    environments = Environment(request.json)
    user_id = get_user_id(request.json)
    apply_terraform(user_id, environments)
    output = get_terraform_output()
    json_output = to_json(output)
    save_ips_for_user(mongo.db, json_output["gci_instances_ids"], user_id)
    return "Created {} environments with requested options. Environments are ready for receiving datasets".format(
        environments.get_nr_instances()
    )


@app.route("/environment/delete", methods=["DELETE"])
def environment_delete():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    destroy_terraform(user_id)
    delete_environment_for_user(mongo.db, environment_id, user_id)
    delete_environment_distribution(mongo.db, environment_id, user_id)
    return "Destroyed environemnt {} for user {}".format(environment_id, user_id)


@app.route("/environment/dataset/data", methods=["POST"])
def environment_dataset_data():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)
    environment_data_distribution = get_environment_data_distribution(
        mongo.db, environment_id, user_id
    )
    for environment_ip, _ in environment_data_distribution["distributions"].items():
        if not (environment_ip in environment["environment_ips"]):
            return (jsonify("Environment ip is invalid"), 400)
    post_data_distribution(
        request.files, environment_data_distribution["distributions"]
    )

    return "Saved data in instances"


@app.route("/environment/dataset/validation", methods=["POST"])
def environment_dataset_validation():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)

    post_data_to_instance(request.files, environment["environment_ips"])
    return "Saved validation data in instances"


@app.route("/environment/dataset/test", methods=["POST"])
def environment_dataset_test():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)

    post_data_to_instance(request.files, environment["environment_ips"])
    return "Saved test data in instances"


@app.route("/environment/dataset/distribution", methods=["POST"])
def environment_dataset_distribution():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    environment = get_environment(mongo.db, environment_id, user_id)
    environment_data_distribution = get_data_distribution(request.json)
    dataset_length = get_dataset_length(request.json)

    for environment_ip, distribution in environment_data_distribution.items():
        if not (environment_ip in environment["environment_ips"]):
            return (jsonify("Environment ip is invalid"), 400)
        environment_data_distribution[environment_ip] = random.sample(
            range(1, dataset_length), distribution
        )
    save_environment_data_distribution(
        mongo.db, environment_id, user_id, environment_data_distribution
    )
    return "Saved dataset distribution"
