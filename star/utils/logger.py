import functools
import logging
import sys

from accelerate.logging import MultiProcessAdapter
from termcolor import colored

__all__ = ["setup_logger"]


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", self._root_name)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def setup_logger(
    name,
    log_level,
    output_file,
):
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name=name,
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return MultiProcessAdapter(logger, {})
