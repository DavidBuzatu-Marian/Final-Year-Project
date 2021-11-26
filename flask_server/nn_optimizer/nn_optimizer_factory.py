import torch
import torch.optim as optim

import sys

sys.path.insert(0, "../../nn_helpers")
from nn_helpers import get_params_from_list


class NNOptimizerFactory:
    def get_optimizer(self, optimizer_type, parameters):
        options = {
            "SGD": self.__build_sgd,
            "RMSProp": self.__build_rmsprop,
            "ASGD": self.__build_asgd,
            "Adamax": self.__build_adamax,
            "Adam": self.__build_adam,
            "Adagrad": self.__build_adagrad,
        }
        if optimizer_type in options:
            return options[optimizer_type](parameters)
        raise Exception("Optimizer type not in options")

    def __build_sgd(self, parameters):
        parameters_set = {"momentum", "weight_decay", "dampening", "nesterov"}
        try:
            params = parameters["params"]
            lr = parameters["lr"]
        except Exception:
            raise Exception("Param or lr is not defined for sgd")
        parameters = get_params_from_list(parameters, parameters_set)
        return optim.SGD(params.parameters(), lr, **parameters)

    def __build_rmsprop(self, parameters):
        parameters_set = {"lr", "momentum", "weight_decay", "alpha", "eps", "centered"}
        try:
            params = parameters["params"]
        except Exception:
            raise Exception("Param or lr is not defined for sgd")
        parameters = get_params_from_list(parameters, parameters_set)
        return optim.RMSProp(params.parameters(), **parameters)

    def __build_asgd(self, parameters):
        parameters_set = {"lr", "lambd", "weight_decay", "alpha", "t0"}
        try:
            params = parameters["params"]
        except Exception:
            raise Exception("Param or lr is not defined for sgd")
        parameters = get_params_from_list(parameters, parameters_set)
        return optim.ASGD(params.parameters(), **parameters)

    def __build_adamax(self, parameters):
        parameters_set = {"lr", "betas", "weight_decay", "eps"}
        try:
            params = parameters["params"]
        except Exception:
            raise Exception("Param or lr is not defined for sgd")
        parameters = get_params_from_list(parameters, parameters_set)
        return optim.Adamax(params.parameters(), **parameters)

    def __build_adam(self, parameters):
        parameters_set = {"lr", "betas", "weight_decay", "eps", "amsgrad"}
        try:
            params = parameters["params"]
        except Exception:
            raise Exception("Param or lr is not defined for sgd")
        parameters = get_params_from_list(parameters, parameters_set)
        return optim.Adam(params.parameters(), **parameters)

    def __build_adagrad(self, parameters):
        parameters_set = {
            "lr",
            "weight_decay",
            "lr_decay",
            "eps",
        }
        try:
            params = parameters["params"]
        except Exception:
            raise Exception("Param or lr is not defined for sgd")
        parameters = get_params_from_list(parameters, parameters_set)
        return optim.Adagrad(params.parameters(), **parameters)
