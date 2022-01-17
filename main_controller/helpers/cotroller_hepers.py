import sys
from logging import error

try:
    from request_helpers import get_to_instance, post_json_to_instance
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


def train_model(instances, training_iterations, instance_training_parameters):
    for _ in range(training_iterations):
        for instance_ip in instances:
            response = post_json_to_instance(
                "http://{}:5000/model/train".format(instance_ip),
                instance_training_parameters,
            )
            with open("model_{}.pth".format(instance_ip), "wb") as instance_model_file:
                instance_model_file.write(response.content)
