import logging
from logging.config import dictConfig


def configure_logging():
    dictConfig({
        'version': 1,
        'formatters': {
            'customFormat': {
                'format': '%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s',
            },
        },
        'handlers': {
            'customFileHandler': {
                'class': 'logging.FileHandler',
                'filename': '../lightning.log',
                'formatter': 'customFormat',
                'level': logging.DEBUG,
            },
        },
        'loggers': {
            'lightning': {
                'handlers': ['customFileHandler'],
                'level': logging.INFO,
                'propagate': 0
            },
        },
    })
