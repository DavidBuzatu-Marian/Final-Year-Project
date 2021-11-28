import torch
import torch.nn as nn

from nn_helpers.nn_helpers import get_params_from_list


class NNLossFactory:
    def get_loss(self, loss_type, parameters):
        options = {
            "L1Loss": self.__build_l1loss,
            "MSELoss": self.__build_mseloss,
            "CrossEntropyLoss": self.__build_crossentropyloss,
            "NLLLoss": self.__build_nllloss,
            "PoissonNLLLoss": self.__build_poissonNLLLoss,
            "GaussianNLLLoss": self.__build_gaussiannllloss,
            "BCELoss": self.__build_bceloss,
            "BCEWithLogitsLoss": self.__build_bcewithlogitsloss,
            "SoftMarginLoss": self.__build_softmarginloss,
            "MultiLabelSoftMarginLoss": self.__build_multilabelsoftmarginloss,
        }
        if loss_type in options:
            return options[loss_type](parameters)
        raise Exception("Loss type not in options")

    def __build_l1loss(self, parameters):
        parameters_set = {"size_average", "reduce", "reduction"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.L1Loss(**parameters)

    def __build_mseloss(self, parameters):
        parameters_set = {"size_average", "reduce", "reduction"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.MSELoss(**parameters)

    def __build_crossentropyloss(self, parameters):
        parameters_set = {
            "weight",
            "size_average",
            "ignore_index",
            "reduce",
            "reduction",
            "label_smoothing",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.CrossEntropyLoss(**parameters)

    def __build_nllloss(self, parameters):
        parameters_set = {
            "weight",
            "size_average",
            "ignore_index",
            "reduce",
            "reduction",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.NLLLoss(**parameters)

    def __build_poissonNLLLoss(self, parameters):
        parameters_set = {
            "log_input",
            "full",
            "size_average",
            "eps",
            "reduce",
            "reduction",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.PoissonNLLLoss(**parameters)

    def __build_gaussiannllloss(self, parameters):
        parameters_set = {
            "full",
            "eps",
            "reduction",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.GaussianNLLLoss(**parameters)

    def __build_bceloss(self, parameters):
        parameters_set = {
            "weight",
            "size_average",
            "reduce",
            "reduction",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.BCELoss(**parameters)

    def __build_bcewithlogitsloss(self, parameters):
        parameters_set = {"weight", "size_average", "reduce", "reduction", "pos_weight"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.BCEWithLogitsLoss(**parameters)

    def __build_softmarginloss(self, parameters):
        parameters_set = {"size_average", "reduce", "reduction"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.SoftMarginLoss(**parameters)

    def __build_multilabelsoftmarginloss(self, parameters):
        parameters_set = {"weight", "size_average", "reduce", "reduction"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.MultiLabelSoftMarginLoss(**parameters)
