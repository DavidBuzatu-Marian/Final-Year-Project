from logging import error, debug
from flask import Flask, request
from flask.json import jsonify
import os
import sys


try:
    from helpers.dataset_helpers import delete_data_from_path
    from helpers.dataset_helpers import delete_model_from_path, save_dataset
    from error_handlers.abort_handler import abort_with_text_response
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))

from app import app


@app.route("/dataset/add", methods=["POST"])
def dataset_add():
    if not ("multipart/form-data" in request.content_type):
        abort_with_text_response(
            415, "Content type is not supported. Please return multipart/form-data")
    if len(request.files) == 0:
        abort_with_text_response(400, "At least a file needs to be selected")
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
        delete_data_from_path(path)
    # delete_model_from_path(os.getenv("MODEL_PATH"))
    return jsonify("Files and model deleted successfully")
