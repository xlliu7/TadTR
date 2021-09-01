# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------


import builtins
import logging
import sys
from .misc import is_main_process


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logger(log_file_path, name=None, level=logging.INFO):
    """
    Setup a logger that simultaneously output to a file and stdout
    ARGS
      log_file_path: string, path to the logging file
    """
    if is_main_process():
        print('this is master process, set up logger')
        # logging settings
        #   log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        log_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(pathname)s: %(lineno)4d: %(message)s",
            datefmt="%m/%d %H:%M:%S")
        root_logger = logging.getLogger(name)
        if name:
            root_logger.propagate = False
        root_logger.setLevel(level)
        # file handler
        if log_file_path is not None:
            log_file_handler = logging.FileHandler(log_file_path)
            log_file_handler.setFormatter(log_formatter)
           
            root_logger.addHandler(log_file_handler)

        # stdout handler
        log_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s]: %(message)s",
            datefmt="%m/%d %H:%M:%S")
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(log_stream_handler)

        logging.info('Log file is %s' % log_file_path)
        return root_logger

    else:
        print('this is not a master process, suppress print')
        _suppress_print()
