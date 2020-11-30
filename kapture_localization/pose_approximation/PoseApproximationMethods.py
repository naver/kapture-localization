# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from enum import auto
from kapture.utils import AutoEnum


class PoseApproximationMethods(AutoEnum):
    equal_weighted_barycenter = auto()
    barycentric_descriptor_interpolation = auto()
    cosine_similarity = auto()

    def __str__(self):
        return self.value


METHOD_DESCRIPTIONS = {
    PoseApproximationMethods.equal_weighted_barycenter: ("EWB: assigns the same weight to all of the top k retrieved "
                                                         "images with w_i = 1/k"),
    PoseApproximationMethods.barycentric_descriptor_interpolation: ("BDI: estimates w_i as the best barycentric "
                                                                    "approximation of the query descriptor via the "
                                                                    "database descriptors with norm2(d_q-sum(w_i*d_i)) "
                                                                    "subject to sum(w_i)=1."),
    PoseApproximationMethods.cosine_similarity: ("CSI: w_i=(1/z_i)*(transpose(d_q)*d_i)^alpha, "
                                                 "z_i=sum(transpose(d_q)*d_j)^alpha")
}
