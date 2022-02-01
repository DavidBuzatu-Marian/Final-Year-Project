from logging import debug, error
from flask import request
import sys
from flask.json import jsonify
import random

from app import app
from app import mongo


@app.route("/health/status", methods=["GET"])
def health_status():
    return "Healthy"
