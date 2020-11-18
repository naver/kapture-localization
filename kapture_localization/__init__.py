# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

# silence kapture_localization logging to critical only, except if told otherwise
import logging
import kapture_localization.utils.path_to_kapture  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
