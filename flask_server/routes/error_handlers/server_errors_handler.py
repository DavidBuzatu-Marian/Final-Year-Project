
import sys
from app import app
import traceback


try:
    from error_handlers.abort_handler import abort_with_text_response
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


def return_500_on_uncaught_server_error(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            exception = log_error()

            abort_with_text_response(500, exception)
    wrapper.__name__ = function.__name__
    return wrapper


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
