import os
from helpers.app_helpers import save_file
import shutil
from logging import error

def delete_model_from_path(path):
    if os.path.isfile(path):
        os.remove(path)


def save_dataset(request):
    data_types = [
        ("train_data", "TRAIN_DATA_PATH"),
        ("train_labels", "TRAIN_LABELS_PATH"),
        ("validation_data", "VALIDATION_DATA_PATH"),
        ("validation_labels", "VALIDATION_LABELS_PATH"),
        ("test_data", "TEST_DATA_PATH"),
        ("test_labels", "TEST_LABELS_PATH"),
    ]
    for key, value in data_types:
        if request.files.getlist(key).length > 0:
            path = os.path.join(os.getenv(value))
            delete_data_from_path(path)
        for file in request.files.getlist(key):
            save_file(file, value)

def delete_data_from_path(path):
    try:
        shutil.rmtree(path)
    except OSError as err:
        error(err)