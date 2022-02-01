from collections import defaultdict
from logging import error
import requests
from flask import abort
import json


def post_to_instance(url, data):
    files = []
    if data.items() == None:
        abort(400, "Posting data went wrong. Empty data")
    for file_type, value in data.items():
        for file in value:
            files.append((file_type, (file.filename, file.read(), file.content_type)))
            file.seek(0)
    if len(files) == 0:
        abort(400, "No files selected")
    response = requests.post(url=url, files=files, timeout=10)
    if not response.ok:
        abort(
            response.status_code,
            "Posting data went wrong. Response: {}".format(response.content),
        )
    return response


def post_json_to_instance(url, json):
    response = requests.post(url, json=json, timeout=10)
    if not response.ok:
        abort(
            response.status_code,
            "Posting data went wrong. Response: {}".format(response.content),
        )
    return response


def get_to_instance(url):
    response = requests.get(url, timeout=10)
    if not response.ok:
        abort(
            response.status_code,
            "Getting from {} went wrong. Response".format(url, response.content),
        )
    return response


def get_instance_data_from_files(files, data_distribution):
    data_keys = [
        "train_data",
        "train_labels",
        "validation_data",
        "validation_labels",
        "test_data",
        "test_labels",
    ]
    instance_data = dict()
    for key in data_keys:
        if len(files.getlist(key)) > 0:
            instance_data[key] = get_instance_data_for_key(
                files.getlist(key), data_distribution
            )
    return instance_data


def get_instance_data_for_key(files, data_distribution):
    data = []
    if len(data_distribution) == 0:
        data = files
    else:
        for i in data_distribution:
            data.append(files[i])
    return data


def post_data_distribution(files, environment_data_distribution):
    for environment_ip, data_distribution in environment_data_distribution.items():
        instance_data = get_instance_data_from_files(files, data_distribution)
        post_to_instance(
            "http://" + environment_ip + ":5000/dataset/add",
            instance_data,
        )


def post_data_to_instance(files, environment_ips):
    instances_data = get_instance_data_from_files(files, [])
    for environment_ip in environment_ips:
        post_to_instance(
            "http://" + environment_ip + ":5000/dataset/add", instances_data
        )


def compute_losses(request_json, environment_ips):
    losses = dict()
    for environment_ip in environment_ips:
        losses[environment_ip] = json.dumps(
            post_json_to_instance(
                "http://" + environment_ip + ":5000/model/loss", request_json
            ).content.decode("utf-8")
        )
    error(losses)
    return losses
