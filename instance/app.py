from flask import Flask
import sys
from dotenv import load_dotenv
import logging

sys.path.insert(0, "./nn_model_factory/model/")
sys.path.insert(1, "./helpers/")
sys.path.insert(2, "./nn_model_factory/")


load_dotenv()

app = Flask(__name__)

logging.basicConfig(
    filename='./logs/general.log', level=logging.INFO,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.basicConfig(
    filename='./logs/error.log', level=logging.ERROR,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.basicConfig(filename='./logs/critical.log', level=logging.CRITICAL,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


import routes.model
import routes.dataset
import routes.instance
