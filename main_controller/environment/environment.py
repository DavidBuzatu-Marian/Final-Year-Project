import numpy as np


class Environment:
    def __init__(self, request_json):
        self.nr_instances = request_json["nr_instances"]
        self.environments = dict()
        for options in request_json["environment_options"]:
            self.environments[options["id"]] = options

    def get_nr_instances(self):
        return self.nr_instances
