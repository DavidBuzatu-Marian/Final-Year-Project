from collections import defaultdict
from logging import error
import requests
from flask import abort


def post_to_instance(url, data):
    files = [
        (file_type, (file.filename, file.read(), file.content_type))
        for file_type, value in data.items()
        for file in value
    ]

    response = requests.post(url=url, files=files)
    if not response.ok:
        abort(400, "Posting data went wrong")
    return response


def get_to_instance(url):
    response = requests.get(url)
    if not response.ok:
        abort(400, "Getting from {} went wrong".format(url))
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
            for i in data_distribution:
                if key in instance_data:
                    instance_data[key].append(files.getlist(key)[i])
                else:
                    instance_data[key] = [files.getlist(key)[i]]
    return instance_data


def post_data_distribution(files, environment_data_distribution):
    for environment_ip, data_distribution in environment_data_distribution.items():
        instance_data = get_instance_data_from_files(files, data_distribution)
        post_to_instance(
            "http://" + environment_ip + ":5000/dataset/add",
            instance_data,
        )
