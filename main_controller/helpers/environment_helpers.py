from logging import error
import subprocess
import json
import os
from bson.objectid import ObjectId
from flask import request
from app import statuses
from datetime import datetime
import sys
import time
from werkzeug.datastructures import ImmutableMultiDict

try:
    from helpers.request_helpers import post_json_to_instance, request_wrapper
    from error_handlers.abort_handler import abort_with_text_response
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def save_ips_for_user(database, ips, environment):
    environment_update = {"environment_ips": [], "status": statuses["1"]}
    environment_query = {
        "user_id": ObjectId(environment.user_id),
        "_id": environment.id}
    for ip in ips["value"]:
        environment_update["environment_ips"].append(ip)
    update_result = database.environmentsAddresses.update_one(
        environment_query, {"$set": environment_update}
    )
    error("Matches: {}. Modified: {}".format(update_result.matched_count, update_result.modified_count))
    return update_result


def send_options_to_instances(ips, environment_options):
    time.sleep(90)  # required for the cold start of docker container
    ips_with_options = set()
    for option in environment_options:
        instance_ip = ips[option['instanceNumber'] - 1]
        request_wrapper(
            lambda:
            post_json_to_instance(
                "http://{}:{}/instance/probabilityoffailure".format(
                    instance_ip,
                    os.getenv("ENVIRONMENTS_PORT")),
                {"probabilityOfFailure": option['probabilityOfFailure']}))
        ips_with_options.add(instance_ip)
    for ip in ips:
        if ip not in ips_with_options:
            request_wrapper(lambda: post_json_to_instance("http://{}:{}/instance/probabilityoffailure"
                                                          .format(ip, os.getenv("ENVIRONMENTS_PORT")),
                                                          {"probabilityOfFailure": 0}))


def update_environment_status(database, environment, status):
    environment_query = {"user_id": ObjectId(environment.user_id), "_id": ObjectId(environment.id)}
    environment_update = {"status": statuses[status]}
    update_result = database.environmentsAddresses.update_one(
        environment_query, {"$set": environment_update}
    )
    return update_result


def delete_environment(database, environment):
    destroy_terraform(environment.user_id)
    delete_environment_for_user(database, environment)
    delete_environment_train_distribution(database, environment)
    delete_environment_data_distribution(database, environment)


def create_environment_data_distribution_entry(database, ips, environment):
    distribution = [{"{}".format(ip): []} for ip in ips["value"]]
    environment_data_distribution_document = {
        "user_id": ObjectId(environment.user_id),
        "environment_id": ObjectId(environment.id),
        "test_data_distribution": distribution,
        "test_labels_data_distribution": distribution,
        "train_data_distribution": distribution,
        "train_labels_data_distribution": distribution,
        "validation_data_distribution": distribution,
        "validation_labels_data_distribution": distribution,
    }
    insert_result = database.environmentsDataDistribution.insert_one(
        environment_data_distribution_document
    )
    return insert_result


def save_environment_for_user(database, user_id, environment):
    environment_document = {
        "user_id": ObjectId(user_id),
        "environment_ips": [],
        "machine_type": environment.machine_type,
        "machine_series": environment.machine_series,
        "status": statuses["0"],
        "environment_options": json.dumps(environment.environment_options),
        "date": int(datetime.utcnow().timestamp()),
    }
    insert_result = database.environmentsAddresses.insert_one(environment_document)
    return insert_result.inserted_id


def delete_environment_for_user(database, environment):
    query = {"_id": ObjectId(environment.id), "user_id": ObjectId(environment.user_id)}
    delete_result = database.environmentsAddresses.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))
    return delete_result


def delete_environment_train_distribution(database, environment):
    query = {"environment_id": ObjectId(environment.id), "user_id": ObjectId(environment.user_id)}
    delete_result = database.environmentsTrainingDataDistribution.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))
    return delete_result


def delete_environment_data_distribution(database, environment):
    query = {"environment_id": ObjectId(environment.id), "user_id": ObjectId(environment.user_id)}
    delete_result = database.environmentsDataDistribution.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))
    return delete_result


def get_environment(database, environment):
    query = {"user_id": ObjectId(environment.user_id), "_id": ObjectId(environment.id)}
    environment = database.environmentsAddresses.find_one(query)
    if environment == None:
        raise ValueError("Environment not found")
    environment["environment_ips"] = set(environment["environment_ips"])
    return environment


def get_environment_data_distribution(database, environment):
    query = {"user_id": ObjectId(environment.user_id), "environment_id": ObjectId(environment.id)}
    data_distribution = database.environmentsTrainingDataDistribution.find_one(query)
    if data_distribution == None:
        raise ValueError("Environment distribution not found")
    return data_distribution


def save_environment_test_data_distribution(
    database, environment, distributions
):
    data_distribution_document = {
        "user_id": ObjectId(environment.user_id),
        "environment_id": ObjectId(environment.id),
        "distributions": distributions,
    }
    data_distribution_query = {
        "user_id": ObjectId(environment.user_id),
        "environment_id": ObjectId(environment.id),
    }
    insert_result = database.environmentsTrainingDataDistribution.update_one(
        data_distribution_query,
        {"$set": data_distribution_document},
        upsert=True
    )
    return insert_result


def save_environment_data_distribution(
    database, environment, distributions
):
    data_distribution_query = {
        "user_id": ObjectId(environment.user_id),
        "environment_id": ObjectId(environment.id),
    }
    update_result = database.environmentsDataDistribution.update_one(
        data_distribution_query, {"$set": distributions}
    )
    return update_result


def get_data_distribution(request_json):
    data_distribution = request_json["data_distribution"]
    return data_distribution


def get_dataset_length(request_json):
    dataset_length = request_json["dataset_length"]
    return dataset_length


def get_user_id(request_json):
    return request_json["user_id"]


def apply_terraform(user_id, environment):
    terraform_apply_result = subprocess.run(
        'cd ./terraform && terraform init && terraform apply -var="nr_instances={}" -var="user_id={}" -auto-approve'.format(
            environment.nr_instances, user_id
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    if terraform_apply_result.returncode != 0:
        destroy_terraform(user_id)
        error("Something went wrong when constructing environment. Error: {}. Return code: {}. Output: {}".format(
            terraform_apply_result.stderr, terraform_apply_result.returncode, terraform_apply_result.stdout, ))
        abort_with_text_response(
            500,
            "Something went wrong when constructing environment. Error: {}. Return code: {}. Output: {}".format(
                terraform_apply_result.stderr,
                terraform_apply_result.returncode,
                terraform_apply_result.stdout,
            ),
        )


def destroy_terraform(user_id):
    terraform_destroy_result = subprocess.run(
        'cd ./terraform && terraform destroy -var="user_id={}" -auto-approve'.format(
            user_id
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    if terraform_destroy_result.returncode != 0:
        abort_with_text_response(
            500,
            "Something went wrong when destroying environments. Error: {}. Return code: {}. Output: {}".format(
                terraform_destroy_result.stderr,
                terraform_destroy_result.returncode,
                terraform_destroy_result.stdout,
            ),
        )


def get_terraform_output():
    output = subprocess.run(
        "cd ./terraform && terraform output -json",
        shell=True,
        capture_output=True,
        text=True,
    )
    if output.returncode != 0:
        abort_with_text_response(
            500,
            "Something went wrong when constructing environments. Error: {}. Return code: {}. Output: {}".format(
                output.stderr,
                output.returncode,
                output.stdout,
            ),
        )
    return output.stdout


# as suggested by: https://stackoverflow.com/a/64187498/11023871
def add_environment_id_to_request(environment_id):
    http_args = request.args.to_dict()
    http_args['environment_id'] = environment_id

    request.args = ImmutableMultiDict(http_args)


def get_environment_id(request_args):
    return ObjectId(request_args.get("environment_id"))


def get_training_options(request_json):
    return request_json["training_options"]


def to_json(object):
    return json.loads(object)
