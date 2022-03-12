

def get_params_from_list(parameters, parameters_set):
    return dict(
        filter(
            lambda key_value: (key_value[0] in parameters_set),
            parameters.items(),
        )
    )
