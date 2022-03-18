import numpy as np


class Environment:
    def __init__(self, request_json):
        self.nr_instances = request_json["nr_instances"]
        self.environment_options = []
        self.machine_type = request_json["machine_type"]
        self.machine_series = request_json["machine_series"]
        for options in request_json["environment_options"]:
            self.environment_options.append(options)
