import flask
from logging import debug, error
import sys
from app import mongo, app
import traceback


try:
    from environment_classes.target_environment import TargetEnvironment
    from helpers.environment_helpers import delete_environment, get_user_id, get_environment_id, update_environment_status
    from error_handlers.abort_handler import abort_with_text_response
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def return_500_on_uncaught_server_error(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            log_error()
            environment = get_environment()
            update_environment_status_to_error(environment, "8")
            abort_with_text_response(500, "Internal Server Error")
    wrapper.__name__ = function.__name__
    return wrapper
# inspired by answer: https://stackoverflow.com/a/53720325/11023871


def return_500_environment_critical_error(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            log_error()
            environment = get_environment()
            update_environment_status_to_error(environment)
            delete_environment(mongo.db, environment)

            abort_with_text_response(500, "Internal Server Error")
    wrapper.__name__ = function.__name__
    return wrapper


def get_environment():
    request_json = flask.request.json
    request_args = flask.request.args
    if "user_id" not in request_json and "user_id" not in request_args:
        return None
    user_id = get_user_id(request_json) if "user_id" in request_json else get_user_id(
        request_args)
    environment_id = get_environment_id(
        request_json) if "environment_id" in request_json else get_environment_id(request_args)
    return TargetEnvironment(user_id, environment_id)


def update_environment_status_to_error(environment, error_code="4"):
    if environment:
        update_environment_status(mongo.db, environment, error_code)

# As suggested by: https://stackoverflow.com/a/49613561/11023871


def log_error():
    exception_type, exception_value, exception_traceback = sys.exc_info()

    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(exception_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" %
                           (trace[0], trace[1], trace[2], trace[3]))

    app.logger.error("Exception type : %s " % exception_type.__name__)
    app.logger.error("Exception message : %s" % exception_value)
    app.logger.error("Stack trace : %s" % stack_trace)

    return exception_value
