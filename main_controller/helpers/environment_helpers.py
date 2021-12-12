from logging import debug
import subprocess
import json
import jsons
from bson.objectid import ObjectId


def save_ips_for_user(database, ips, user_id):
    db = database
    environments_document = {"user_id": user_id, "environment_ips": []}
    for ip in ips["value"]:
        environments_document["environment_ips"].append(ip)
    insert_result = db.environments_addresses.insert_one(environments_document)
    debug("Created entry: {}".format(insert_result.inserted_id))
    return insert_result.inserted_id


def delete_environment_for_user(database, environment_id, user_id):
    db = database
    query = {"_id": ObjectId(environment_id), "user_id": user_id}
    debug(query)
    delete_result = db.environments_addresses.delete_one(query)
    debug("Deleted entry: {}".format(delete_result.deleted_count))


# TODO: Get user id from auhentication token
def get_user_id(request_json):
    return request_json["user_id"]


def apply_terraform(environments):
    terraform_apply_result = subprocess.run(
        'cd ./terraform && terraform apply -var="nr_instances={}" -auto-approve'.format(
            environments.get_nr_instances()
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    if terraform_apply_result.returncode != 0:
        return (
            "Something went wrong when constructing environments. Check logs for more details",
            500,
        )


def destroy_terraform():
    terraform_destroy_result = subprocess.run(
        "cd ./terraform && terraform destroy -auto-approve",
        shell=True,
        capture_output=True,
        text=True,
    )
    if terraform_destroy_result.returncode != 0:
        return (
            "Something went wrong when destroying environments. Check logs for more details",
            500,
        )


def get_terraform_output():
    output = subprocess.run(
        "cd ./terraform && terraform output -json",
        shell=True,
        capture_output=True,
        text=True,
    )
    if output.returncode != 0:
        return (
            "Something went wrong when getting outputs. Check logs for more details",
            500,
        )
    return output.stdout


def get_environment_id(request_json):
    return request_json["environment_id"]


def to_json(object):
    return json.loads(object)
