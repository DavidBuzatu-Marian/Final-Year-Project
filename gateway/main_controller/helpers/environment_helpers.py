from logging import debug, error
import subprocess
import json
import jsons
from bson.objectid import ObjectId
from flask import abort


def save_ips_for_user(database, ips, user_id):
    db = database
    environments_document = {"user_id": user_id, "environment_ips": []}
    for ip in ips["value"]:
        environments_document["environment_ips"].append(ip)
    insert_result = db.environments_addresses.insert_one(environments_document)
    error("Created entry: {}".format(insert_result.inserted_id))
    return insert_result.inserted_id


def delete_environment_for_user(database, environment_id, user_id):
    db = database
    query = {"_id": ObjectId(environment_id), "user_id": user_id}
    delete_result = db.environments_addresses.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))


def delete_environment_distribution(database, environment_id, user_id):
    db = database
    query = {"_id": ObjectId(environment_id), "user_id": user_id}
    delete_result = db.environments_data_distribution.delete_one(query)
    error("Deleted entry: {}".format(delete_result.deleted_count))


def get_environment(database, environment_id, user_id):
    db = database
    query = {"user_id": user_id, "_id": ObjectId(environment_id)}
    environment = db.environments_addresses.find_one(query)
    if environment == None:
        raise ValueError("Environment not found")
    environment["environment_ips"] = set(environment["environment_ips"])
    return environment


def get_environment_data_distribution(database, environment_id, user_id):
    db = database
    query = {"user_id": user_id, "_id": ObjectId(environment_id)}
    data_distribution = db.environments_data_distribution.find_one(query)
    if data_distribution == None:
        raise ValueError("Environment distribution not found")
    return data_distribution


def save_environment_data_distribution(
    database, environment_id, user_id, distributions
):
    data_distribution_document = {
        "user_id": user_id,
        "_id": ObjectId(environment_id),
        "distributions": distributions,
    }
    insert_result = database.environments_data_distribution.insert_one(
        data_distribution_document
    )
    return insert_result.inserted_id


def get_data_distribution(request_json):
    data_distribution = request_json["data_distribution"]
    return data_distribution


def get_dataset_length(request_json):
    dataset_length = request_json["dataset_length"]
    return dataset_length


# TODO: Get user id from auhentication token
def get_user_id(request_json):
    return int(request_json["user_id"])


def apply_terraform(user_id, environments):
    terraform_apply_result = subprocess.run(
        'cd ./terraform && terraform apply -var="nr_instances={}" -var="user_id={}" -auto-approve'.format(
            environments.get_nr_instances(), user_id
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    if terraform_apply_result.returncode != 0:
        destroy_terraform(user_id)
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
    return request_json["environment_id"]


def to_json(object):
    return json.loads(object)