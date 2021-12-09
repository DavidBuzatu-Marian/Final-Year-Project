from logging import error
from flask import Flask, request
import sys
from flask.json import jsonify
import torch
import jsons
from app import app
from app import mongo

from environment import Environment


@app.route("/environment/create", methods=["POST"])
def environment_create():
    environments = Environment(request.json)
    error(environments)
    return jsons.dump(environments)
