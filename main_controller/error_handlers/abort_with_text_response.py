from urllib import response
from flask import abort, make_response

# Inspired by: https://stackoverflow.com/a/69829766/11023871


def abort_with_text_response(status_code, message):
    response = make_response("Status code: {}\nMessage: {}".format(status_code, message))
    response.status_code = status_code
    abort(response)
