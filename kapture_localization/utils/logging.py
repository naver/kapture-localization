# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Logging specific to kapture_localization
"""

import logging

logging.basicConfig(format='%(levelname)-8s::%(name)s: %(message)s')


def getLogger():
    """
    Get the default kapture_localization logger.

    :return: logger
    """
    return logging.getLogger('kapture_localization')
