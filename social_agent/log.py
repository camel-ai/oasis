import logging
import colorlog
import datetime

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

def my_log(log_file = r'D:\study\GWorks\Tasks\Agent\24-04-22-code\agent.log'):
    logging.Formatter.converter = beijing
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    logger = logging.getLogger('logger_name')
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


class Log:
    logger = my_log()

    @staticmethod
    def info(msg):
        Log.logger.info(msg)

    @staticmethod
    def debug(msg):
        Log.logger.debug(msg)

    @staticmethod
    def warning(msg):
        Log.logger.warning(msg)

    @staticmethod
    def error(msg):
        Log.logger.error(msg)

    @staticmethod
    def critical(msg):
        Log.logger.critical(msg)
