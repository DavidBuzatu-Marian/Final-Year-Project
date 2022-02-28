import sys
import os
import torch
from logging import error
from flask.helpers import send_file
from werkzeug.datastructures import FileStorage
from flask import abort

try:
    from request_helpers import get_to_instance, post_json_to_instance, post_to_instance
    from nn_model import NNModel
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def get_available_instances(environment, max_trials, required_instances):
    trials = 0
    available_instances = set()
    while trials < max_trials and len(available_instances) < required_instances:
        # Get each environment ip
        # Make request for availability and store available envs
        for environment_ip in environment["environment_ips"]:
            response = get_to_instance(
                "http://{}:5000/instance/availability".format(environment_ip)
            )
            if response.json()["availability"] == True:
                available_instances.add(environment_ip)
        trials += 1
    return list(available_instances)


def get_training_iterations(request_json):
    return request_json["training_iterations"]


def get_instance_training_parameters(request_json):
    return request_json["environment_parameters"]


def get_model_network_options(request_json):
    return request_json["environment_model_network_options"]


def train_model(instances, training_iterations, instance_training_parameters):
    for _ in range(training_iterations):
        train_on_instances(instances, instance_training_parameters)
        if len(instances) == 0 :
            abort(400, "All devices crashed")
        aggregated_model = aggregate_models(instances)
        save_aggregated_model(aggregated_model)
        update_instances_model(instances)
    return send_file(os.getenv("GLOBAL_MODEL"))


def update_instances_model(instances):
    model_file = open(os.getenv("GLOBAL_MODEL"), "rb")
    model = FileStorage(model_file)
    for instance_ip in instances:
        post_to_instance(
            "http://{}:5000/model/update".format(instance_ip),
            {"model": [model]},
        )


def save_aggregated_model(aggregated_model):
    torch.save(aggregated_model, os.getenv("GLOBAL_MODEL"))


def aggregate_models(instances):
    aggregated_model = None
    for instance_ip in instances:
        model = load_model_from_path("./models/model_{}.pth".format(instance_ip))
        if aggregated_model == None:
            aggregated_model = model
        else:
            state_aggregated = aggregated_model.state_dict()
            state_model = model.state_dict()
            for layer in state_aggregated:
                state_aggregated[layer] += state_model[layer]
            aggregated_model.load_state_dict(state_aggregated)
        delete_model_from_path("./models/model_{}.pth".format(instance_ip))
    state_aggregated = aggregated_model.state_dict()
    for layer in state_aggregated:
        state_aggregated[layer] /= len(instances)
    aggregated_model.load_state_dict(state_aggregated)
    return aggregated_model


def train_on_instances(instances, instance_training_parameters):
    for instance_ip in instances:
        response = post_json_to_instance(
            "http://{}:5000/model/train".format(instance_ip),
            instance_training_parameters,
            True
        )
        if not response.ok:
            # Instance failed during training => remove instance from round of training
            instances.remove(instance_ip)
        else:
            with open(
            "./models/model_{}.pth".format(instance_ip), "wb"
            ) as instance_model_file:
                instance_model_file.write(response.content)


def load_model_from_path(path):
    if os.path.isfile(path):
        return torch.load(path)
    else:
        raise FileNotFoundError("Model not found")


def delete_model_from_path(path):
    if os.path.isfile(path):
        os.remove(path)
    else:
        raise FileNotFoundError("Model not found: {}".format(path))


def create_model(environment_ips, model_network_options):
    for instance_ip in environment_ips:
        post_json_to_instance(
            "http://{}:5000/model/create".format(instance_ip), model_network_options
        )
