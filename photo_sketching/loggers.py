# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A module with logging functions.
"""
import os
import numpy as np
import logging
import warnings
import pandas as pd
import time
from collections import OrderedDict
from string import Formatter

logger = logging.getLogger()

class ConsoleFormatter(logging.Formatter):

    def __init__(self, log_format = "%(name)s: %(levelname)s - %(message)s"):
        super(ConsoleFormatter, self).__init__()
        bright_black = "\033[0;90m"
        bright_white = "\033[0;97m"
        yellow = "\033[0;33m"
        red = "\033[0;31m"
        magenta = "\033[0;35m"
        white = "\033[0;37m"

        self.log_formats = {logging.DEBUG: bright_black + log_format + bright_white,
                            logging.INFO: white + log_format + bright_white,
                            logging.WARNING: yellow + log_format + bright_white,
                            logging.ERROR: red + log_format + bright_white,
                            logging.CRITICAL: magenta + log_format + bright_white}

    def format(self, record):
        log_fmt = self.log_formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging(level="info", logfile=None):
    """ Setup the logging.

    Parameters
    ----------
    level : str, default "info"
        the logger level
    logfile: str, default None
        the log file.
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    level = LEVELS.get(level, None)
    if level is None:
        raise ValueError("Unknown logging level.")
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(stream_handler)
    if logfile is not None:
        logging_format = logging.Formatter(
            "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - "
            "%(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    if level != logging.DEBUG:
        warnings.simplefilter("ignore", DeprecationWarning)

def setup_logging_updated(level="info", logfile=None):
    """ Setup the logging.

    Parameters
    ----------
    level : str, default "info"
        the logger level
    logfile: str, default None
        the log file.
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    level = LEVELS.get(level, None)
    if level is None:
        raise ValueError("Unknown logging level.")
    
    handlers = []
    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(ConsoleFormatter())
    handlers.append(stream_handler)
    # File Handler
    if logfile is not None:
        logging_format = logging.Formatter(
            "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - "
            "%(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_format)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        handlers=handlers
    )


class TrainLogger(logging.Logger):

    def __init__(self, name):
        super().__init__(name=name)
        self.history = OrderedDict()
        self.current_step = -1
    
    def store(self, **kwargs):
        if self.current_step not in self.history.keys():
            self.history[self.current_step] = {}
        for key, value in kwargs.items():
            if key in self.history[self.current_step].keys():
                if not isinstance(self.history[self.current_step][key], list):
                    self.history[self.current_step][key] = [self.history[self.current_step][key]]
                self.history[self.current_step][key].append(value)
            else:
                self.history[self.current_step][key] = value

    def reduce(self, reduce_fx="sum"):
        for key, val in self.history[self.current_step].items():
            if isinstance(val, list):
                if reduce_fx == "mean":
                    self.history[self.current_step][key] = np.mean(val)
                elif reduce_fx == "max":
                    self.history[self.current_step][key] = np.max(val)
                elif reduce_fx == "min":
                    self.history[self.current_step][key] = np.min(val)
                elif reduce_fx == "sum":
                    self.history[self.current_step][key] =  np.sum(val)
    
    def step(self):
        self.current_step += 1
        self.history[self.current_step] = {"timestep": time.time()}

    def summary(self, keys=None):
        msg = ""
        if keys is not None:
            for key in keys:
                value = self.history[self.current_step][key]
                msg += f"{key}: {value:.2g} | "
        else:        
            fre = ("fold", "run", "epoch")
            msg = " - ".join([m for m in [f"{k.capitalize()}: {self.history[self.current_step].get(k)}" for k in fre] if "None" not in m])
            for key, value in self.history[self.current_step].items():
                if key not in fre and key != "timestep":
                    msg += f" | {key}: {value}"
            msg += f" | {self.get_duration(formatting='{h}h {min}min')}"
        self.info(msg)
        return msg
    
    def get_current_step(self):
        return self.current_step

    def reset_history(self):
        self.history = OrderedDict()
        self.current_step = -1
    
    def get_duration(self, formatting="{h}h {min}min {s}s"):
        duration = time.time() - next(iter(self.history.values()), {"timestep": 0})["timestep"]
        fieldnames = [fname for _, fname, _, _ in Formatter().parse(formatting) if fname]
        timer = {}
        if "h" in fieldnames:
            timer["h"] = int(duration // 3600)
            duration = duration % 3600
        if "min" in fieldnames:
            timer["min"] = int(duration // 60)
            duration = duration % 60
        if "s" in fieldnames:
            timer["s"] = int(duration)
        return formatting.format(**timer)

    def save(self, chkpt_dir, filename=""):
        dico2save = OrderedDict()
        # create columns and row rows
        for step, dico in self.history.items():
            for key, value in dico.items():
                if key not in dico2save.keys():
                    dico2save[key] = [None for _ in range(step)]
                elif len(dico2save[key]) < step:
                    to_fill = step - len(dico2save[key])
                    dico2save[key].extend([None for _ in range(to_fill)])
                dico2save[key].append(value)
        # complete rows
        for key, values in dico2save.items():
            if len(values)-1 < self.current_step:
                to_fill = self.current_step - len(values) + 1
                dico2save[key].extend([None for _ in range(to_fill)])
        # reorder columns
        for key in ("epoch", "run", "fold"):
            if key in dico2save.keys():
                filename = f"_{key}-{dico2save[key][-1]}{filename}"
                dico2save.move_to_end(key, last=False)
        dico2save.move_to_end("timestep", last=True)
        filename = f"{self.name}{filename}.csv"
        df2save = pd.DataFrame(dico2save)
        df2save.to_csv(os.path.join(chkpt_dir, filename), sep=",", index=False)

if __name__ == "__main__":
    setup_logging_updated(level="info", 
                          logfile="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching/test/logs.log")
    #logging.setLoggerClass(TrainLogger)
    #history = logging.getLogger('DebugLogger')
    logger.info('logger.info')
    history = TrainLogger("myname")
    for epoch in range(0, 50, 5):
        history.step()
        history.store(epoch=epoch)
        if epoch % 2 == 0:
            history.store(rmse=3)
            history.store(rmse=2)
            history.store(rmse=1)
        else:
            history.store(roc_auc=2)
            history.store(roc_auc=5)
        if epoch % 5 == 0:
                history.summary()
        history.reduce("mean")
        history.info("My name is Bond !")
    print(history.get_current_step())
    history.info("message info history")
    history.warning("ceci est un message history")
    print("level", history.level)
    history.save("/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching/test")