"""
Logger Utils

Modified from mmcv

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging

import torch
import torch.distributed as dist
from termcolor import colored

logger_initialized = {}
root_status = 0


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="a", color=False):
    """Initialize and get a logger by name."""
    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    plain_formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message."""
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )


def get_root_logger(log_file=None, log_level=logging.INFO, file_mode="a"):
    """Get the Pointcept root logger."""
    return get_logger(
        name="pointcept",
        log_file=log_file,
        log_level=log_level,
        file_mode=file_mode,
    )


def _log_api_usage(identifier: str):
    """Log usage of Pointcept components through PyTorch's internal API."""
    torch._C._log_api_usage_once("pointcept." + identifier)
