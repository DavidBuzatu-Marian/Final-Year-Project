from logging import error, debug
from flask import Flask, request
from flask.json import jsonify
import os
from helpers.app_helpers import read_model_from_path, save_file
import json
import shutil

from app import app


@app.route("/dataset/add", methods=["POST"])
def dataset_add():
    if not ("multipart/form-data" in request.content_type):
        return (
            jsonify("Content type is not supported. Please return multipart/form-data"),
            415,
        )
    if len(request.files) == 0:
        return (jsonify("At least a file needs to be selected"), 400)
    for file in request.files.getlist("train"):
        save_file(file, "TRAIN_DATA_PATH")
    for file in request.files.getlist("validation"):
        save_file(file, "VALIDATION_DATA_PATH")
    for file in request.files.getlist("test"):
        save_file(file, "TEST_DATA_PATH")
    return jsonify("Files save successfully")


@app.route("/dataset/remove", methods=["POST"])
def dataset_remove():
    dir_paths = [
        os.path.join(os.getenv("TRAIN_DATA_PATH")),
        os.path.join(os.getenv("VALIDATION_DATA_PATH")),
        os.path.join(os.getenv("TEST_DATA_PATH")),
    ]
    for path in dir_paths:
        try:
            shutil.rmtree(path)
        except OSError as err:
            error(err)
    return jsonify("Files deleted successfully")
