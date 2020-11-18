# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import numpy as np
import cvxpy as cp
from .PoseApproximationMethods import PoseApproximationMethods
from kapture_localization.image_retrieval.pairing import StackedGlobalFeatures, get_similarity_matrix
from kapture_localization.utils.logging import getLogger


def get_interpolation_weights(method: PoseApproximationMethods,
                              query_gfeat: StackedGlobalFeatures,
                              map_gfeat: StackedGlobalFeatures,
                              topk: int,
                              additional_parameters: dict):
    """
    compute the pose weights for the given method as a dict { query image name -> list(map image name, weight) }  

    :param method: pose approximation method to use
    :type method: PoseApproximationMethods
    :param query_gfeat: global features for the query images
    :type query_gfeat: StackedGlobalFeatures
    :param map_gfeat: global features for the map images
    :type map_gfeat: StackedGlobalFeatures
    :param topk: the max number of top retained images
    :type topk: int
    :param additional_parameters: method specific parameters
    :type additional_parameters: dict
    """
    similarity_matrix = get_similarity_matrix(query_gfeat, map_gfeat)
    local_topk = min(topk, similarity_matrix.shape[1])
    if local_topk != topk:
        getLogger().warning(f'topk was set to {local_topk} instead of {topk} because there were not enough map data')
    similarity_sorted = np.empty((similarity_matrix.shape[0], local_topk), dtype=int)
    for i, scores in enumerate(similarity_matrix):
        indexes = np.argsort(-scores)
        similarity_sorted[i, :] = indexes[:local_topk]

    if method == PoseApproximationMethods.equal_weighted_barycenter:
        weights = _get_EWB_weights(similarity_matrix.shape[0], local_topk)
    elif method == PoseApproximationMethods.barycentric_descriptor_interpolation:
        weights = _get_BDI_weights(similarity_sorted, query_gfeat, map_gfeat)
    elif method == PoseApproximationMethods.cosine_similarity:
        assert 'alpha' in additional_parameters
        weights = _get_CSI_weights(similarity_matrix, similarity_sorted, additional_parameters['alpha'])
    else:
        raise NotImplementedError(f'{method} - unknown PoseApproximationMethods')

    weights_dict = {}
    for i, indexes in enumerate(similarity_sorted):
        query_name = query_gfeat.index[i]
        weights_dict[query_name] = list(zip(map_gfeat.index[indexes], weights[i, :]))
    return weights_dict


def _get_EWB_weights(number_of_queries: int, topk: int):
    """
    get equal weighted barycenter weights
    """
    weights = np.zeros((number_of_queries, topk))
    weights[:, :] = 1.0 / topk
    return weights


def _get_BDI_weights(similarity_sorted: np.ndarray,
                     query_gfeat: StackedGlobalFeatures,
                     map_gfeat: StackedGlobalFeatures):
    """
    barycentric descriptor interpolation : interpolating baseline of http://openaccess.thecvf.com/content_CVPR_2019/papers/Sattler_Understanding_the_Limitations_of_CNN-Based_Absolute_Camera_Pose_Regression_CVPR_2019_paper.pdf
    """
    np.random.seed(0)
    weights = np.zeros(similarity_sorted.shape)
    topk = similarity_sorted.shape[1]
    for i in range(similarity_sorted.shape[0]):
        query_descriptor = query_gfeat.stacked_features[i]
        interpolating_descriptors = map_gfeat.stacked_features[similarity_sorted[i]]

        A = interpolating_descriptors.T
        b = query_descriptor

        w = cp.Variable(topk)
        objective = cp.Minimize(cp.sum_squares(A@w - b))
        constraints = [cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights[i, :] = w.value
    return weights


def _get_CSI_weights(similarity_matrix: np.ndarray,
                     similarity_sorted: np.ndarray,
                     alpha: float):
    """
    cosine similarity
    """
    weights = np.zeros(similarity_sorted.shape)
    for i in range(similarity_sorted.shape[0]):
        weights[i, :] = similarity_matrix[i, similarity_sorted[i, :]]**(alpha)
        weights[i, :] = weights[i, :] / np.sum(weights[i, :])
    return weights
