import logging
import logging.config
import traceback
import os

class LogHelper():
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
            },
            'console': {
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            },
            'file': {
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            }
        },
        'handlers': {
            'console':{
                'level':'DEBUG',
                'class':'logging.StreamHandler',
                'formatter': 'console'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'file',
                'filename': 'log/log.txt',
                'mode': 'a',
                'maxBytes': 10485760,
                'backupCount': 2,
            }
        },
        'loggers': {
            'root': {
                'handlers':['console', 'file'],
                'propagate': True,
                'level':'DEBUG',
            }
        }
    }

    os.makedirs('log', exist_ok=True)
    logging.config.dictConfig(LOGGING)
    logger = logging.getLogger('root')

    def debug(msg):
        LogHelper.logger.debug(msg)
    @staticmethod
    def info(msg):
        LogHelper.logger.info(msg)
    def warning(msg):
        LogHelper.logger.warning(msg)
    def error(msg=None):
        if msg:
            LogHelper.logger.error(msg + '\n' + traceback.format_exc())
        else:
            LogHelper.logger.error(traceback.format_exc())
    def critical(msg):
        LogHelper.logger.critical(msg)
