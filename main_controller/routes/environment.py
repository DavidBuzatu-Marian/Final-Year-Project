from logging import error
from flask import Flask, request
import sys
from flask.json import jsonify
import torch
import jsons
import subprocess

from app import app
from app import mongo

from environment import Environment


@app.route("/environment/create", methods=["POST"])
def environment_create():
    environments = Environment(request.json)
    result = subprocess.run(
        'cd ./terraform && terraform apply -var="instance_number=2" -auto-approve',
        shell=True,
        capture_output=True,
        text=True,
    )
    return jsonify(jsons.dump({result.stdout, result.stderr, result.returncode}))
