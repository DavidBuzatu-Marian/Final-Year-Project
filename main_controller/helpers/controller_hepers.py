import sys
import os
import torch
from logging import error
from flask.helpers import send_file
from werkzeug.datastructures import FileStorage
from bson.objectid import ObjectId

try:
    from request_helpers import get_to_instance, post_json_to_instance, post_to_instance, request_wrapper
    from nn_model import NNModel
    from error_handlers.abort_handler import abort_with_text_response
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def get_available_instances(environment, max_trials, required_instances):
    trials = 0
    available_instances = set()
    while trials < max_trials and len(available_instances) < required_instances:
        # Get each environment ip
        # Make request for availability and store available envs
        for environment_ip in environment["environment_ips"]:
            response = request_wrapper(lambda: get_to_instance(
                "http://{}:5000/instance/availability".format(environment_ip)
            ))
            if response.json()["availability"] == True:
                available_instances.add(environment_ip)
        trials += 1
    if len(available_instances) < required_instances:
        abort_with_text_response(400, "Not enough available instances found")
    return available_instances


def get_training_iterations(request_json):
    return request_json["training_iterations"]


def get_instance_training_parameters(request_json):
    return request_json["environment_parameters"]


def get_model_network_options(request_json):
    return request_json["environment_model_network_options"]


def train_model(database, instances, training_iterations, instance_training_parameters, user_id,
                environment_id):
    initial_instances = instances
    train_log = list()
    for iteration in range(training_iterations):
        train_on_instances(instances, instance_training_parameters)
        write_to_train_log(train_log, process_training_results(
            iteration, instances, initial_instances))
        if len(instances) == 0:
            write_to_train_log(train_log, ["All devices crashed"])
            write_logs_to_database(database, train_log, user_id, environment_id)
            abort_with_text_response(400, "All devices crashed")
        aggregated_model = aggregate_models(instances, environment_id, user_id)
        write_to_train_log(train_log, ["Aggregated models from contributors"])
        save_aggregated_model(aggregated_model, environment_id, user_id)
        update_instances_model(instances, environment_id, user_id)
        write_to_train_log(
            train_log,
            ["Updated model on contributors. Sent aggregated model to intances",
             "Preparing next round..."])
    write_logs_to_database(database, train_log, user_id, environment_id)
    return send_file(os.getenv("GLOBAL_MODEL"))


def write_logs_to_database(database, logs, user_id, environment_id):
    log_document = {
        "user_id": ObjectId(user_id),
        "environment_id": ObjectId(environment_id),
        "train_logs": logs
    }
    log_document_query = {
        "user_id": ObjectId(user_id),
        "environment_id": ObjectId(environment_id),
    }
    insert_result = database.environmentsLogs.update_one(
        log_document_query, {"$set": log_document}, upsert=True)
    return insert_result


def process_training_results(iteration, instances, initial_instances):
    data = list()
    data.append("Iteration nr: %d" % (iteration))
    for instance_ip in initial_instances:
        training_result = check_contribution(instance_ip, instances)
        data.append("Instance IP: %s , Training result: %s , Other: None" % (
            instance_ip, training_result))
    return data


def check_contribution(instance_ip, instances):
    if instance_ip in instances:
        return "Contributed to current training round"
    else:
        return "Does not contribute to training process anymore"


def write_to_train_log(train_log, data):
    for log in data:
        train_log.append(log)


def update_instances_model(instances):
    model_file = open(os.getenv("GLOBAL_MODEL"), "rb")
    model = FileStorage(model_file)
    for instance_ip in instances:
        post_to_instance(
            "http://{}:5000/model/update".format(instance_ip),
            {"model": [model]},
        )


def save_aggregated_model(aggregated_model, environment_id, user_id):
    torch.save(aggregated_model, "{}/{}-{}.pth".format(os.getenv("GLOBAL_MODEL"), environment_id, user_id))


def aggregate_models(instances, environment_id, user_id):
    aggregated_model = None
    for instance_ip in instances:
        model = load_model_from_path(
            "./models/model-{}-{}-{}.pth".format(environment_id, user_id, instance_ip))
        if aggregated_model == None:
            aggregated_model = model
        else:
            state_aggregated = aggregated_model.state_dict()
            state_model = model.state_dict()
            for layer in state_aggregated:
                state_aggregated[layer] += state_model[layer]
            aggregated_model.load_state_dict(state_aggregated)
        delete_model_from_path(
            "./models/model-{}-{}-{}.pth".format(environment_id, user_id, instance_ip))
    state_aggregated = aggregated_model.state_dict()
    for layer in state_aggregated:
        state_aggregated[layer] /= len(instances)
    aggregated_model.load_state_dict(state_aggregated)
    return aggregated_model


def train_on_instances(instances, instance_training_parameters, environment_id, user_id):
    for instance_ip in instances:
        response = request_wrapper(lambda: post_json_to_instance(
            "http://{}:5000/model/train".format(instance_ip),
            instance_training_parameters,
            True
        ))
        if not response.ok:
            # Instance failed during training => remove instance from round of training
            instances.remove(instance_ip)
        else:
            with open(
                "./models/model-{}-{}-{}.pth".format(environment_id, user_id, instance_ip), "wb"
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
            "http://{}:5000/model/create".format(
                instance_ip), model_network_options
        )
