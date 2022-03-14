from collections import defaultdict
from logging import error
import requests
import json
import sys


try:
    from error_handlers.abort_handler import abort_with_text_response
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def request_wrapper(request_function):
    try:
        return request_function()
    except requests.exceptions.Timeout:
        abort_with_text_response(408, "A request to an instance timedout.")
    except requests.exceptions.RequestException as e:
        abort_with_text_response(500, "A request failed due to an internal server error")
    except ValueError:
        abort_with_text_response(
            500, "A request failed due to an internal server error (ValueError)")


def post_to_instance(url, data):
    files = []
    if data.items() == None:
        abort_with_text_response(400, "Posting data went wrong. Empty data")
    for file_type, value in data.items():
        for file in value:
            files.append((file_type, (file.filename, file.read(), file.content_type)))
            file.seek(0)
    if len(files) == 0:
        abort_with_text_response(400, "No files selected")
    response = requests.post(url=url, files=files, timeout=10)
    if not response.ok:
        abort_with_text_response(
            response.status_code,
            "Posting data went wrong. Response: {}".format(response.content),
        )
    return response


def post_json_to_instance(url, json, allow_failure=False):
    response = requests.post(url, json=json, timeout=10)
    if not response.ok and not allow_failure:
        abort_with_text_response(
            response.status_code,
            "Posting data went wrong. Response: {}".format(response.content),
        )
    return response


def get_to_instance(url):
    response = requests.get(url, timeout=10)
    if not response.ok:
        abort_with_text_response(
            response.status_code,
            "Getting from: {} went wrong. Response: {}".format(url, response.content),
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
    instances_data = dict()
    for environment_ip, data_distribution in environment_data_distribution.items():
        instance_data = get_instance_data_from_files(files, data_distribution)
        request_wrapper(lambda: post_to_instance(
            "http://" + environment_ip + ":5000/dataset/add",
            instance_data,
        ))
        add_data_to_instance_distribution(instances_data, instance_data, environment_ip)
    return instances_data


def add_data_to_instance_distribution(instances_data, instance_data, environment_ip):
    data_keys = {
        "train_data": "train_data_distribution",
        "train_labels": "train_labels_data_distribution",
        "validation_data": "validation_data_distribution",
        "validation_labels": "validation_labels_data_distribution",
        "test_data": "test_data_distribution",
        "test_labels": "test_labels_data_distribution",
    }
    for filetype, instance_files in instance_data.items():
        filenames = []
        for file in instance_files:
            filenames.append(file.filename)
        if data_keys[filetype] in instances_data:
            instances_data[data_keys[filetype]].append({environment_ip: filenames})
        else:
            instances_data[data_keys[filetype]] = [{environment_ip: filenames}]


def post_data_to_instance(files, environment_ips):
    instances_data = dict()
    instance_data = get_instance_data_from_files(files, [])
    for environment_ip in environment_ips:
        request_wrapper(lambda: post_to_instance(
            "http://" + environment_ip + ":5000/dataset/add", instance_data
        ))
        add_data_to_instance_distribution(instances_data, instance_data, environment_ip)
    return instances_data


def compute_losses(request_json, environment_ips):
    losses = dict()
    for environment_ip in environment_ips:
        losses[environment_ip] = json.dumps(
            request_wrapper(lambda: post_json_to_instance(
                "http://" + environment_ip + ":5000/model/loss", request_json
            )).content.decode("utf-8")
        )
    error(losses)
    return losses
