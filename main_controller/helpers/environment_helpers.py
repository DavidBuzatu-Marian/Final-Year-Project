from logging import debug, error
import subprocess
import json
from bson.objectid import ObjectId
from flask import abort
from app import statuses
from datetime import datetime


def save_ips_for_user(database, ips, user_id, environment_id):
    environment_update = {"environment_ips": [], "status": statuses["1"]}
    environment_query = {"user_id": ObjectId(user_id), "_id": environment_id}
    for ip in ips["value"]:
        environment_update["environment_ips"].append(ip)
    update_result = database.environmentsAddresses.update_one(
        environment_query, {"$set": environment_update}
    )
    error("Mathces: {}. Modified: {}".format(update_result.matched_count, update_result.modified_count))
    return update_result


def update_environment_status(database, user_id, environment_id, status):
    environment_query = {"user_id": ObjectId(user_id), "_id": ObjectId(environment_id)}
    environment_update = {"status": statuses[status]}
    update_result = database.environmentsAddresses.update_one(
        environment_query, {"$set": environment_update}
    )
    return update_result


def create_environment_data_distribution_entry(database, ips, user_id, environment_id):
    distribution = [{"{}".format(ip): []} for ip in ips["value"]]
    error(distribution)
    environment_data_distribution_document = {
        "user_id": ObjectId(user_id),
        "environment_id": ObjectId(environment_id),
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
        "machine_type": environment.get_machine_type(),
        "machine_series": environment.get_machine_series(),
        "status": statuses["0"],
        "environment_options": json.dumps(environment.get_environment_options()),
        "date": int(datetime.utcnow().timestamp()),
    }
    insert_result = database.environmentsAddresses.insert_one(environment_document)
    return insert_result.inserted_id


def delete_environment_for_user(database, environment_id, user_id):
    query = {"_id": ObjectId(environment_id), "user_id": ObjectId(user_id)}
    delete_result = database.environmentsAddresses.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))
    return delete_result


def delete_environment_train_distribution(database, environment_id, user_id):
    query = {"environment_id": ObjectId(environment_id), "user_id": ObjectId(user_id)}
    delete_result = database.environmentsTrainingDataDistribution.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))
    return delete_result


def delete_environment_data_distribution(database, environment_id, user_id):
    query = {"environment_id": ObjectId(environment_id), "user_id": ObjectId(user_id)}
    delete_result = database.environmentsDataDistribution.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))
    return delete_result


def get_environment(database, environment_id, user_id):
    query = {"user_id": ObjectId(user_id), "_id": ObjectId(environment_id)}
    environment = database.environmentsAddresses.find_one(query)
    if environment == None:
        raise ValueError("Environment not found")
    environment["environment_ips"] = set(environment["environment_ips"])
    return environment


def get_environment_data_distribution(database, environment_id, user_id):
    query = {"user_id": ObjectId(user_id), "environment_id": ObjectId(environment_id)}
    data_distribution = database.environmentsTrainingDataDistribution.find_one(query)
    if data_distribution == None:
        raise ValueError("Environment distribution not found")
    return data_distribution


def save_environment_test_data_distribution(
    database, environment_id, user_id, distributions
):
    data_distribution_document = {
        "user_id": ObjectId(user_id),
        "environment_id": ObjectId(environment_id),
        "distributions": distributions,
    }
    insert_result = database.environmentsTrainingDataDistribution.insert_one(
        data_distribution_document
    )
    return insert_result.inserted_id


def save_environment_data_distribution(
    database, user_id, environment_id, distributions
):
    data_distribution_query = {
        "user_id": ObjectId(user_id),
        "environment_id": ObjectId(environment_id),
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


# TODO: Get user id from auhentication token
def get_user_id(request_json):
    return request_json["user_id"]


def apply_terraform(user_id, environments):
    terraform_apply_result = subprocess.run(
        'cd ./terraform && terraform init && terraform apply -var="nr_instances={}" -var="user_id={}" -auto-approve'.format(
            environments.get_nr_instances(), user_id
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    if terraform_apply_result.returncode != 0:
        destroy_terraform(user_id)
        error( "Something went wrong when constructing environments. Error: {}. Return code: {}. Output: {}".format(
                terraform_apply_result.stderr,
                terraform_apply_result.returncode,
                terraform_apply_result.stdout,
            ))
        abort(
            500,
            "Something went wrong when constructing environments. Error: {}. Return code: {}. Output: {}".format(
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
        abort(
            500,
            "Something went wrong when constructing environments. Error: {}. Return code: {}. Output: {}".format(
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
        abort(
            500,
            "Something went wrong when constructing environments. Error: {}. Return code: {}. Output: {}".format(
                output.stderr,
                output.returncode,
                output.stdout,
            ),
        )
    return output.stdout


def get_environment_id(request_json):
    return ObjectId(request_json["environment_id"])


def to_json(object):
    return json.loads(object)
