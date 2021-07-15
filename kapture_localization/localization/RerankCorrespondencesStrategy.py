# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
from enum import auto
import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils import AutoEnum


class RerankCorrespondencesStrategy(AutoEnum):
    none = auto()
    matches_count = auto()
    correspondences_count = auto()

    def __str__(self):
        return self.value
