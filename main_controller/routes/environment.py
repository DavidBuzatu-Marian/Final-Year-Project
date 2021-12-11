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
    environment_ip = save_ips_for_user(
        mongo.db, json_output["ec2_instances_public_ip"], user_id
    )
    return "Created {} environments with requested options. Environments are ready for receiving datasets".format(
        environment_ip
    )
