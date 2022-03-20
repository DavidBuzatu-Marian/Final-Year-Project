from logging import error
from flask import Flask, request
from flask.helpers import send_file
from flask.json import jsonify
import json
import os
import sys
import torch


try:
    from nn_model_factory.nn_model import NNModel
    from helpers.data_helpers import *
    from helpers.app_helpers import *
    from helpers.model_helpers import is_failing
    from error_handlers.abort_handler import abort_with_text_response
    from routes.error_handlers.server_errors_handler import return_500_on_uncaught_server_error
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))

from app import app


@app.route("/model/create", methods=["POST"])
@return_500_on_uncaught_server_error
def model_create():
    model = NNModel(request.json)
    path = os.getenv("MODEL_PATH")
    if os.path.isfile(path):
        os.remove(path)
    torch.save(model, path)
    return jsonify(repr(model.nn_layers))


@app.route("/model/loss", methods=["POST"])
@return_500_on_uncaught_server_error
def model_validate():
    path = os.getenv("MODEL_PATH")
    model = read_model_from_path(path)
    loss_func = get_loss(request.json)
    hyperparameters = get_hyperparameters(request.json)

    loss_type = get_loss_type(request.json)

    validation_dataloader = get_dataloader(
        data_path=os.getenv(loss_type["data_path"]),
        labels_path=os.getenv(loss_type["labels_path"]),
        hyperparameters=hyperparameters,
    )

    total_loss = 0
    mean, std = compute_mean_and_std(validation_dataloader)
    with torch.no_grad():
        for data, label in validation_dataloader:
            data = reshape_data(data, hyperparameters)
            if standardize(hyperparameters):
                data = standardize_data(data, mean, std)
            if normalize(hyperparameters):
                data = normalize_data(
                    data, hyperparameters["data_min"],
                    hyperparameters["data_max"])

            output = model(data)

            loss = loss_func(output, label.to(torch.int64))
            total_loss = loss.item()

    return json.dumps({"total_loss": total_loss / len(validation_dataloader)})


@app.route("/model/train", methods=["POST"])
@return_500_on_uncaught_server_error
def model_train():
    path = os.getenv("MODEL_PATH")
    model = read_model_from_path(path)
    loss_func = get_loss(request.json)
    optimizer = get_optimizer(request.json, model)
    hyperparameters = get_hyperparameters(request.json)
    processors = get_processors(request.json)
    probability_of_failure = get_instance_probability_of_failure()

    train_dataloader = get_dataloader(
        data_path=os.getenv("TRAIN_DATA_PATH"),
        labels_path=os.getenv("TRAIN_LABELS_PATH"),
        hyperparameters=hyperparameters,
    )
    mean, std = compute_mean_and_std(train_dataloader)
    for _ in range(0, hyperparameters["epochs"]):
        if is_failing(probability_of_failure):
            abort_with_text_response(400, "Device failed during training")
        model.train()
        for data, label in train_dataloader:
            optimizer.zero_grad()

            data = reshape_data(data, hyperparameters)
            if normalize(hyperparameters):
                data = normalize_data(data, mean, std)
            output = model(data)

            output = process_output(output, processors)
            loss = loss_func(output, label.to(dtype=torch.int64))
            loss.backward()
            optimizer.step()

    torch.save(model, path)
    return send_file(os.getenv("MODEL_PATH"))


@app.route("/model/update", methods=["POST"])
def model_update():
    request.files["model"].save(os.getenv("MODEL_PATH"))
    return "Saved aggregated model"
