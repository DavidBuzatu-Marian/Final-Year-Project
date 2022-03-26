from http import client
import json
import sys
import os
import torch
from logging import error
from flask.helpers import send_file
from werkzeug.datastructures import FileStorage
from bson.objectid import ObjectId
from flask import copy_current_request_context
import asyncio

import random
import concurrent.futures

try:
    from request_helpers import get_to_instance, post_json_to_instance, post_to_instance, request_wrapper
    from environment_helpers import update_environment_status
    from error_handlers.abort_handler import abort_with_text_response
    from nn_model_factory.nn_model import NNModel

except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def get_available_instances(environment_ips, max_trials, required_instances):
    trials = 0
    available_instances = set()
    while trials < max_trials and len(available_instances) < required_instances:
        # Get each environment ip
        # Make request for availability and store available envs
        for environment_ip in environment_ips:
            if environment_ip not in available_instances:
                response = request_wrapper(lambda: get_to_instance(
                    "http://{}:{}/instance/availability".format(environment_ip, os.getenv("ENVIRONMENTS_PORT")), allow_failure=True))
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


def train_model(database, environment_ips, training_options, training_iterations,
                instance_training_parameters, environment):
    train_log = list()
    write_logs_to_database(database, train_log, environment)  # used to reset logs
    for iteration in range(training_iterations):
        instances = get_random_available_instances(environment_ips, training_options)
        initial_instances = instances.copy()

        instances_error = train_on_instances(instances, instance_training_parameters, environment)
        write_to_train_log(train_log, process_training_results(
            iteration, instances, initial_instances, instances_error))
        if len(instances) == 0:
            write_to_train_log(train_log, ["All devices crashed"])
            write_logs_to_database(database, train_log, environment)
            update_environment_status(database, environment, "8")
            abort_with_text_response(400, "All devices crashed")
        aggregated_model = aggregate_models(instances, environment)
        write_to_train_log(train_log, ["Aggregated models from contributors"])
        save_aggregated_model(aggregated_model, environment)
        update_instances_model(instances, environment)
        write_to_train_log(
            train_log,
            ["Updated model on contributors. Sent aggregated model to intances",
             "Preparing next round..."])
    write_logs_to_database(database, train_log, environment)
    update_environment_status(database, environment, "7")
    return send_file("{}/{}-{}.pth".format(os.getenv("GLOBAL_MODEL"),
                                           environment.id, environment.user_id))


def get_random_available_instances(environment_ips, training_options):
    available_instances = get_available_instances(
        environment_ips, training_options['max_trials'], training_options['required_instances']
    )
    return random.sample(list(available_instances), training_options['required_instances'])


def write_logs_to_database(database, logs, environment):
    log_document = {
        "user_id": ObjectId(environment.user_id),
        "environment_id": ObjectId(environment.id),
        "train_logs": logs
    }
    log_document_query = {
        "user_id": ObjectId(environment.user_id),
        "environment_id": ObjectId(environment.id),
    }
    insert_result = database.environmentsLogs.update_one(
        log_document_query, {"$set": log_document}, upsert=True)
    return insert_result


def process_training_results(iteration, instances, initial_instances, instances_error):
    data = list()
    data.append("Iteration nr: %d" % (iteration))
    for instance_ip in initial_instances:
        training_result = check_contribution(instance_ip, instances)
        detected_error = get_error(instance_ip, instances, instances_error)
        data.append("Instance IP: %s , Training result: %s , Errors: %s" % (
            instance_ip, training_result, detected_error))
    return data


def get_error(instance_ip, instances, instances_error):
    if instance_ip not in instances:
        return instances_error[instance_ip]
    return "No error detected"


def check_contribution(instance_ip, instances):
    if instance_ip in instances:
        return "Contributed to current training round"
    else:
        return "Does not contribute to training process anymore"


def write_to_train_log(train_log, data):
    for log in data:
        train_log.append(log)


def update_instances_model(instances, environment):
    model_file = open("{}/{}-{}.pth".format(os.getenv("GLOBAL_MODEL"),
                                            environment.id, environment.user_id), "rb")
    model = FileStorage(model_file)
    for instance_ip in instances:
        request_wrapper(lambda: post_to_instance(
            "http://{}:{}/model/update".format(instance_ip, os.getenv("ENVIRONMENTS_PORT")),
            {"model": [model]}, timeout=100
        ))


def save_aggregated_model(aggregated_model, environment):
    torch.save(aggregated_model, "{}/{}-{}.pth".format(os.getenv("GLOBAL_MODEL"),
               environment.id, environment.user_id))


def aggregate_models(instances, environment):
    aggregated_model = None
    for instance_ip in instances:
        model = load_model_from_path(
            "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, instance_ip))
        if aggregated_model == None:
            aggregated_model = model
        else:
            state_aggregated = aggregated_model.state_dict()
            state_model = model.state_dict()
            for layer in state_aggregated:
                state_aggregated[layer] += state_model[layer]
            aggregated_model.load_state_dict(state_aggregated)
        delete_model_from_path(
            "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, instance_ip))
    state_aggregated = aggregated_model.state_dict()
    for layer in state_aggregated:
        state_aggregated[layer] /= len(instances)
    aggregated_model.load_state_dict(state_aggregated)
    return aggregated_model


def train_on_instances(instances, json_parameters, environment):
    instances_error = dict()

    def post_requests(instances, json_parameters, environment):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            @copy_current_request_context
            def fetch(instance_ip):
                url = "http://{}:{}/model/train".format(instance_ip, os.getenv("ENVIRONMENTS_PORT"))
                response = request_wrapper(lambda: post_json_to_instance(
                    url, json_parameters, allow_failure=True, timeout=20000))
                return response
            future_to_training_responses = {executor.submit(
                fetch, instance_ip): instance_ip for instance_ip in instances}
            process_futures(future_to_training_responses, instances_error, instances, environment)

    post_requests(instances, json_parameters, environment)
    return instances_error


def process_futures(future_to_training_responses, instances_error, instances, environment):
    for future in concurrent.futures.as_completed(future_to_training_responses):
        instance_ip = future_to_training_responses[future]
        try:
            response = future.result()
        except Exception as exc:
            error(exc)
            instances_error[instance_ip] = exc
            instances.remove(instance_ip)
        else:
            process_training_response(
                response, instances, instances_error, instance_ip, environment)


def process_training_response(response, instances, instances_error, instance_ip, environment):
    if not response.ok:
        # Instance failed during training => remove instance from round of training
        instances_error[instance_ip] = response.text
        instances.remove(instance_ip)
    else:
        os.makedirs('./models', exist_ok=True)
        with open(
                "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, instance_ip), "wb+") as instance_model_file:
            instance_model_file.write(response.content)


# Deprecated
# def train_on_instances(instances, instance_training_parameters, environment):
#     instances_error = dict()
#     for instance_ip in instances:
#         response = request_wrapper(lambda: post_json_to_instance(
#             "http://{}:{}/model/train".format(instance_ip, os.getenv("ENVIRONMENTS_PORT")),
#             instance_training_parameters,
#             True,
#             10000
#         ))
#         if not response.ok:
#             # Instance failed during training => remove instance from round of training
#             instances_error[instance_ip] = response.text
#             instances.remove(instance_ip)
#         else:
#             with open(
#                 "./models/model-{}-{}-{}.pth".format(environment.id,
#                                                      environment.user_id, instance_ip), "wb"
#             ) as instance_model_file:
#                 instance_model_file.write(response.content)
#     return instances_error


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
        request_wrapper(lambda: post_json_to_instance(
            "http://{}:{}/model/create".format(
                instance_ip, os.getenv("ENVIRONMENTS_PORT")), model_network_options, timeout=20))
