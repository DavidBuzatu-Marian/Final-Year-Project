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
    from routes.error_handlers.environment import return_500_environment_create_error
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


@app.route("/environment/create", methods=["POST"])
@return_500_environment_create_error
def environment_create():
    environment = Environment(request.json)
    user_id = get_user_id(request.json)
    environment_id = save_environment_for_user(mongo.db, user_id, environment)
    add_environment_id_to_request(environment_id)
    apply_terraform(user_id, environment)
    output = get_terraform_output()
    json_output = to_json(output)
    save_ips_for_user(
        mongo.db, json_output["gci_instances_ids"], user_id, environment_id
    )
    send_options_to_instances(json_output["gci_instances_ids"]["value"], environment.get_environment_options())
    create_environment_data_distribution_entry(
        mongo.db,
        json_output["gci_instances_ids"],
        user_id,
        environment_id,
    )
    return "Created {} environments with requested options. Environments are ready for receiving datasets".format(
        environment.get_nr_instances()
    )


@app.route("/environment/delete", methods=["DELETE"])
def environment_delete():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    update_environment_status(mongo.db, user_id, environment_id, "6")
    delete_environment(mongo.db, user_id, environment_id)
    return "Destroyed environment {} for user {}".format(environment_id, user_id)


@app.route("/environment/dataset/data", methods=["POST"])
def environment_dataset_data():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)
    update_environment_status(mongo.db, user_id, environment_id, "2")
    environment_data_distribution = get_environment_data_distribution(
        mongo.db, environment_id, user_id
    )
    for environment_ip, _ in environment_data_distribution["distributions"].items():
        if not (environment_ip in environment["environment_ips"]):
            return (jsonify("Environment ip is invalid"), 400)
    instances_data = post_data_distribution(
        request.files, environment_data_distribution["distributions"]
    )
    save_environment_data_distribution(
        mongo.db, user_id, environment_id, instances_data
    )
    update_environment_status(mongo.db, user_id, environment_id, "5")
    return "Saved data in instances"


@app.route("/environment/dataset/validation", methods=["POST"])
def environment_dataset_validation():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)
    update_environment_status(mongo.db, user_id, environment_id, "2")
    instances_data = post_data_to_instance(
        request.files, environment["environment_ips"]
    )
    save_environment_data_distribution(
        mongo.db, user_id, environment_id, instances_data
    )
    update_environment_status(mongo.db, user_id, environment_id, "5")
    return "Saved validation data in instances"


@app.route("/environment/dataset/test", methods=["POST"])
def environment_dataset_test():
    user_id = get_user_id(request.args)
    environment_id = get_environment_id(request.args)
    environment = get_environment(mongo.db, environment_id, user_id)
    update_environment_status(mongo.db, user_id, environment_id, "2")
    instances_data = post_data_to_instance(
        request.files, environment["environment_ips"]
    )
    save_environment_data_distribution(
        mongo.db, user_id, environment_id, instances_data
    )
    update_environment_status(mongo.db, user_id, environment_id, "5")
    return "Saved test data in instances"


@app.route("/environment/dataset/distribution", methods=["POST"])
def environment_dataset_distribution():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    environment = get_environment(mongo.db, environment_id, user_id)
    environment_data_distribution = get_data_distribution(request.json)
    dataset_length = get_dataset_length(request.json)

    # TODO: Choose between random distribution or given distribution
    for environment_ip, distribution in environment_data_distribution.items():
        if not (environment_ip in environment["environment_ips"]):
            return (jsonify("Environment ip is invalid"), 400)
        environment_data_distribution[environment_ip] = random.sample(
            range(1, dataset_length), distribution
        )
    save_environment_test_data_distribution(
        mongo.db, environment_id, user_id, environment_data_distribution
    )
    return "Saved dataset distribution"
