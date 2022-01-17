import sys
from logging import error

try:
    from request_helpers import get_to_instance
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
            if response.json()["availability"]:
                available_instances.add(environment_ip)
        trials += 1
    return list(available_instances)
