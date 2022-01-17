import sys

try:
    from request_helpers import get_to_instance
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def get_available_instances(environment, max_trials, required_instances):
    trials = 0
    available_instances = []
    while trials < max_trials and len(available_instances) < required_instances:
        # Get each environment ip
        # Make request for availability and store available envs
        for environment_ip in environment["environment_ips"]:
            response = get_to_instance(environment_ip)
            if response["availability"]:
                available_instances.append(environment_ip)
        trials += 1
    return available_instances
