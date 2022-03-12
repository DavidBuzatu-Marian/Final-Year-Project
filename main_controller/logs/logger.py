import logging

logging.basicConfig(
    filename='general.log', level=logging.INFO,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.basicConfig(
    filename='error.log', level=logging.ERROR,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.basicConfig(filename='critical.log', level=logging.CRITICAL,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
