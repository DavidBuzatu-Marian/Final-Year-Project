import flask
from logging import debug, error
import sys
from app import mongo
import traceback

try:
    from helpers.environment_helpers import delete_environment, get_user_id, get_environment_id, update_environment_status
except ImportError as exc:
     sys.stderr.write("Error: failed to import modules ({})".format(exc))



# inspired by answer: https://stackoverflow.com/a/53720325/11023871
def return_500_environment_create_error(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            print_error()
            request_json = flask.request.json
            user_id = get_user_id(request_json)

            environment_id = get_environment_id(flask.request.args)
            update_environment_status(mongo.db, user_id, environment_id, "4")
            delete_environment(mongo.db, user_id, environment_id)
            response = {
                'status_code': 500,
                'status': 'Internal Server Error'
            }
            return flask.jsonify(response), 500
    return wrapper

# As suggested by: https://stackoverflow.com/a/49613561/11023871
def print_error():
    ex_type, ex_value, ex_traceback = sys.exc_info()

    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

    error("Exception type : %s " % ex_type.__name__)
    error("Exception message : %s" %ex_value)
    error("Stack trace : %s" %stack_trace)