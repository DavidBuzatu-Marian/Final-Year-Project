from logging import error
import numpy as np

def is_failing(probability_of_failure):
    sample_space = np.linspace(0, 1, 1000)
    choice = np.random.choice(sample_space)
    if choice > probability_of_failure:
        return False
    return True