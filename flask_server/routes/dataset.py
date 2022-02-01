from logging import error, debug
from flask import Flask, request
from flask.json import jsonify
import os
import sys

try:
    from helpers.dataset_helpers import delete_model_from_path, save_dataset
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))
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
    save_dataset(request)
    # delete_model_from_path(os.getenv("MODEL_PATH"))
    return jsonify("Files saved successfully")


@app.route("/dataset/remove", methods=["POST"])
def dataset_remove():
    dir_paths = [
        os.path.join(os.getenv("TRAIN_DATA_PATH")),
        os.path.join(os.getenv("TRAIN_LABELS_PATH")),
        os.path.join(os.getenv("VALIDATION_DATA_PATH")),
        os.path.join(os.getenv("VALIDATION_LABELS_PATH")),
        os.path.join(os.getenv("TEST_DATA_PATH")),
        os.path.join(os.getenv("TEST_LABELS_PATH")),
    ]
    for path in dir_paths:
        try:
            shutil.rmtree(path)
        except OSError as err:
            error(err)
    # delete_model_from_path(os.getenv("MODEL_PATH"))
    return jsonify("Files and model deleted successfully")
