# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Logging specific to kapture_localization
"""

import logging
import os
import json
import pathlib

logging.basicConfig(format='%(levelname)-8s::%(name)s: %(message)s')


def getLogger():
    """
    Get the default kapture_localization logger.

    :return: logger
    """
    return logging.getLogger('kapture_localization')


def save_to_json(arguments_as_dict: dict, filepath: str):
    """
    save dict to json
    """
    p = pathlib.Path(filepath)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(arguments_as_dict, f, default=str)


def load_json(filepath: str):
    """
    load dict from json
    """
    with open(filepath, 'r') as f:
        return json.load(f)
