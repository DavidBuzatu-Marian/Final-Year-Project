from logging import debug, error
from flask import Flask, request
import sys
from flask.json import jsonify
import torch
import jsons
import subprocess
import json

from app import app
from app import mongo

from environment import Environment


@app.route("/environment/create", methods=["POST"])
def environment_create():
    environments = Environment(request.json)
    # TODO: Get user id from auhentication token
    user_id = request.json["user_id"]
    result = subprocess.run(
        'cd ./terraform && terraform apply -var="nr_instances={}" -auto-approve'.format(
            environments.get_nr_instances()
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (
            500,
            "Something went wrong when constructing environments. Check logs for more details",
        )
    output = subprocess.run(
        "cd ./terraform && terraform output -json",
        shell=True,
        capture_output=True,
        text=True,
    )
    json_output = json.loads(output.stdout)
    environment_ip = save_ips_for_user(json_output["ec2_instances_public_ip"], user_id)
    return "Created {} environments with requested options. Environments are ready for receiving datasets".format(
        environment_ip
    )


def save_ips_for_user(ips, user_id):
    db = mongo.db
    environments_document = {"user_id": user_id, "environment_ips": []}
    for ip in ips["value"]:
        environments_document["environment_ips"].append(ip)
    insert_result = db.environments_addresses.insert_one(environments_document)
    debug("Created entry: {}".format(insert_result.inserted_id))
    return insert_result.inserted_id
