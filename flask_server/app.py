from flask import Flask
import sys
from dotenv import load_dotenv

sys.path.insert(0, "./nn_model/")


load_dotenv()

app = Flask(__name__)

import routes.model
import routes.dataset
import routes.instance
