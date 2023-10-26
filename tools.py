import logging


def get_handler(log_file='LLaVA_model.log', log_level=None):
    if log_level is None:log_level = logging.INFO
    f_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s:%(lineno)s] - %(message)s')
    f_handler.setFormatter(formatter)
    f_handler.setLevel(log_level)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(f_handler)
    return logger
