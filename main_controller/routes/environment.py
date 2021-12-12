from logging import debug, error
from flask import Flask, request
import sys
from flask.json import jsonify
import torch

from app import app
from app import mongo

from environment import Environment
from helpers.environment_helpers import *


@app.route("/environment/create", methods=["POST"])
def environment_create():
    environments = Environment(request.json)
    user_id = get_user_id(request.json)
    apply_terraform(environments)
    output = get_terraform_output()
    json_output = to_json(output)
    save_ips_for_user(mongo.db, json_output["ec2_instances_public_ip"], user_id)
    return "Created {} environments with requested options. Environments are ready for receiving datasets".format(
        environments.get_nr_instances()
    )


@app.route("/environment/delete", methods=["POST"])
def environment_delete():
    user_id = get_user_id(request.json)
    environment_id = get_environment_id(request.json)
    destroy_terraform()
    delete_environment_for_user(mongo.db, environment_id, user_id)
    return "Destroyed environemnt {} for user {}".format(environment_id, user_id)
