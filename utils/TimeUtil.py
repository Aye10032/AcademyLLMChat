import time
from loguru import logger


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f'{func.__name__} Time cost: {end - start}s')
        return result

    return wrapper
