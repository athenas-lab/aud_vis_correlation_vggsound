#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
utility functions for A-V correlation model.
"""
import logging

""" Config utilities """

def getDataStr(data_split):
        tr   = data_split["train"]
        val  = data_split["val"]
        test = data_split["test"]
        tr_str = "tr_%s"%("_".join(tr))
        val_str = "val_%s"%("_".join(val))
        #test_str = "test_%s"%("_".join(test))
        #model depends on train and val set. 
        data_str = "_".join([tr_str, val_str])
        print(data_str)
        return data_str

def getTagStr(data):
    """ Convert tag description to string """
    
    data_str = "_".join([data["level"], "tag", data["src"], data["feat"]])
    return data_str

def getVfeatStr(data):
    """ Convert visual feat description to string """
    
    data_str = "_".join([data["type"], str(data["fps"])])+"fps"
    return data_str
    
def setLogger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. 
    """
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        #stream_handler = logging.StreamHandler()
        #stream_handler.setFormatter(logging.Formatter('%(message)s'))
        #logger.addHandler(stream_handler)
    return

