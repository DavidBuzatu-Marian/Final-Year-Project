import numpy as np


class Environment:
    def __init__(self, request_json):
        self.nr_instances = request_json["nr_instances"]
        self.environment_options = []
        self.machine_type = request_json["machine_type"]
        for options in request_json["environment_options"]:
            self.environment_options.append(options)

    def get_nr_instances(self):
        return self.nr_instances

    def get_machine_type(self):
        return self.machine_type

    def get_environment_options(self):
        return self.environment_options
