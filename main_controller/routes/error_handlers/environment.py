import flask
from logging import debug, error
import sys
from app import mongo

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