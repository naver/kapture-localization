# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
from enum import auto
import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils import AutoEnum


class DuplicateCorrespondencesStrategy(AutoEnum):
    keep = auto()
    ignore_strict = auto()  # same kpid / ptid
    ignore = auto()  # same kpid or ptid
    ignore_same_kpid = auto()
    ignore_same_p3did = auto()

    def __str__(self):
        return self.value
