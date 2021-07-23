# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
from enum import auto
from typing import List, Optional, Dict, Tuple
import numpy as np

import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils import AutoEnum


class LateFusionMethod(AutoEnum):
    mean = auto()
    power = auto()
    maximum = auto()
    minimum = auto()
    mean_and_power = auto()
    min_and_max = auto()
    villena_roman = auto()
    mean_and_min = auto()
    harmonic_mean = auto()
    generalized_harmonic_mean = auto()
    round_robin = auto()

    def __str__(self):
        return self.value


# An empirical study of fusion operators for multimodal image retrieval
# Gabriela Csurka and Stéphane Clinchant
# Xerox Research Center Europe, Meylan, France
METHOD_DESCRIPTIONS = {
    LateFusionMethod.mean: "sum(alpha_i * sim_i) where sum(alpha_i) = 1",
    LateFusionMethod.power: "product(sim_i^alpha_i) where sum(alpha_i) = 1",
    LateFusionMethod.maximum: "max(sim_i)",
    LateFusionMethod.minimum: "min(sim_i)",
    LateFusionMethod.mean_and_power: ("gamma*sum(alpha_i * sim_i)+(1-gamma)*product(sim_i^alpha_i) "
                                      "where sum(alpha_i) = 1, 0<=gamma<=1"),
    LateFusionMethod.min_and_max: "alpha*min(sim_i)+(1-alpha)*max(sim_i) where 0<=alpha<=1",
    LateFusionMethod.villena_roman: ("(1-alpha)*min(sim_i)+alpha*(min(sim_i)^2/(max(sim_i)+min(sim_i))) "
                                     "where 0<=alpha<=1"),
    LateFusionMethod.mean_and_min: ("gamma*sum(alpha_i * sim_i)+(1-gamma)*min(sim_i) "
                                    "where sum(alpha_i) = 1, 0<=gamma<=1"),
    LateFusionMethod.harmonic_mean: "f^-1(sum(f(x_i)/n)) where f=1/x, x_i=alpha_i*sim_i, sum(alpha_i) = 1",
    LateFusionMethod.generalized_harmonic_mean: ("f^-1(sum(f(x_i)/n)) where "
                                                 "f=1/(gamma+x), x_i=alpha_i*sim_i, sum(alpha_i) = 1, gamma>=0"),
    LateFusionMethod.round_robin: "Select the top image from each global feature, then the second best.. until topk"
}


def _get_normalized_weights(weights: Optional[List[float]], number_of_elements: int) -> List[float]:
    if weights is None or len(weights) == 0:
        weights = [1.0 for _ in range(number_of_elements)]

    assert len(weights) == number_of_elements
    sum_weights = sum(weights)
    assert sum_weights > 0
    return [v/sum_weights for v in weights]


def fuse_similarities(similarity_matrices: List[np.ndarray],
                      method: LateFusionMethod,
                      method_dependent_parameters: dict):
    """
    fuse similarity matrices with given method, this method doesn't work with round robin
    """
    similarity = None
    if method == LateFusionMethod.mean:
        weights = _get_normalized_weights(method_dependent_parameters["weights"], len(similarity_matrices))
        similarity = np.sum([v * mat for mat, v in zip(similarity_matrices, weights)], axis=0)
    elif method == LateFusionMethod.power:
        weights = _get_normalized_weights(method_dependent_parameters["weights"], len(similarity_matrices))
        similarity = np.ones(similarity_matrices[0].shape, dtype=similarity_matrices[0].dtype)
        for sim_matrix, weight in zip(similarity_matrices, weights):
            similarity = np.multiply(similarity, np.power(sim_matrix, weight))
    elif method == LateFusionMethod.maximum:
        similarity = np.max(similarity_matrices, axis=0)
    elif method == LateFusionMethod.minimum:
        similarity = np.min(similarity_matrices, axis=0)
    elif method == LateFusionMethod.mean_and_power:
        weights = _get_normalized_weights(method_dependent_parameters["weights"], len(similarity_matrices))
        gamma = method_dependent_parameters['gamma']
        assert gamma >= 0 and gamma <= 1
        mean = np.sum([v * mat for mat, v in zip(similarity_matrices, weights)], axis=0)
        power = np.ones(similarity_matrices[0].shape, dtype=similarity_matrices[0].dtype)
        for sim_matrix, weight in zip(similarity_matrices, weights):
            power = np.multiply(power, np.power(sim_matrix, weight))
        similarity = gamma * mean + (1 - gamma) * power
    elif method == LateFusionMethod.min_and_max:
        min_weight = method_dependent_parameters['min_weight']
        max_weight = method_dependent_parameters['max_weight']
        sum_weight = min_weight + max_weight
        assert sum_weight > 0
        similarity = np.sum([min_weight/sum_weight * np.min(similarity_matrices, axis=0),
                             max_weight/sum_weight * np.max(similarity_matrices, axis=0)], axis=0)
    elif method == LateFusionMethod.villena_roman:
        alpha = method_dependent_parameters['alpha']
        assert alpha >= 0 and alpha <= 1
        max_mat = np.max(similarity_matrices, axis=0)
        min_mat = np.min(similarity_matrices, axis=0)
        left_side = (1-alpha) * max_mat
        denom = min_mat + max_mat
        right_side = alpha * np.multiply(np.power(min_mat, 2.0), np.reciprocal(denom))
        similarity = left_side + right_side
    elif method == LateFusionMethod.mean_and_min:
        weights = _get_normalized_weights(method_dependent_parameters["weights"], len(similarity_matrices))
        gamma = method_dependent_parameters['gamma']
        assert gamma >= 0 and gamma <= 1
        mean = np.sum([v * mat for mat, v in zip(similarity_matrices, weights)], axis=0)
        min_mat = np.min(similarity_matrices, axis=0)
        similarity = gamma * mean + (1 - gamma) * min_mat
    elif method == LateFusionMethod.harmonic_mean:
        n = len(similarity_matrices)
        weights = _get_normalized_weights(method_dependent_parameters["weights"], n)
        similarity = np.reciprocal(np.sum([(1/n) * np.reciprocal(v * mat)
                                           for mat, v in zip(similarity_matrices, weights)], axis=0))
    elif method == LateFusionMethod.generalized_harmonic_mean:
        n = len(similarity_matrices)
        weights = _get_normalized_weights(method_dependent_parameters["weights"], n)
        gamma = method_dependent_parameters['gamma']
        assert gamma >= 0
        gamma_mat = gamma * np.ones(similarity_matrices[0].shape, similarity_matrices[0].dtype)
        similarity = np.reciprocal(np.sum([(1/n) * np.reciprocal(gamma_mat + (v * mat))
                                           for mat, v in zip(similarity_matrices, weights)], axis=0)) - gamma_mat
    else:
        raise NotImplementedError(f'late fusion method {method.name} is not supported by this function')
    return similarity


def round_robin_from_similarity_dicts(similarity_dicts: List[Dict[str, List[Tuple[str, float]]]],
                                      topk=Optional[int]):
    """
    fuse sorted similarity_dicts using round robin
    """
    image_pairs = []
    query_list = set().union(*[k.keys() for k in similarity_dicts])
    for query_name in sorted(query_list):
        k = 0
        added_images = set()
        if topk is None:
            map_images = set()
            for similarity_dict_j in similarity_dicts:
                if query_name not in similarity_dict_j:
                    continue
                similarity_dict_j_map_images = (v[0] for v in similarity_dict_j[query_name])
                map_images.update(similarity_dict_j_map_images)
            local_topk = len(map_images)
        else:
            local_topk = topk
        for i in range(local_topk):
            if k >= local_topk:
                break
            for similarity_dict_j in similarity_dicts:
                if query_name not in similarity_dict_j:
                    continue
                if i >= len(similarity_dict_j[query_name]):
                    continue
                map_name, _ = similarity_dict_j[query_name][i]
                if map_name not in added_images:
                    image_pairs.append((query_name, map_name, local_topk - k))
                    added_images.add(map_name)
                    k += 1
                    if k >= local_topk:
                        break
    return image_pairs


def get_image_retrieval_late_fusion_argparser(method: LateFusionMethod):
    """
    return the per method arguments for the late fusion
    """
    parser_method = argparse.ArgumentParser(description=METHOD_DESCRIPTIONS[method],
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_method.set_defaults(method=method)
    # per method parameters
    if method == LateFusionMethod.mean or method == LateFusionMethod.power or method == LateFusionMethod.harmonic_mean:
        parser_method.add_argument('--weights', nargs='+', default=[], type=float, help='Weights')
    elif method == LateFusionMethod.min_and_max:
        parser_method.add_argument('--min-weight',
                                   default=0.5,
                                   type=float,
                                   help='weight for min(similarities)')
        parser_method.add_argument('--max-weight',
                                   default=0.5,
                                   type=float,
                                   help='weight for max(similarities)')
    elif method == LateFusionMethod.villena_roman:
        parser_method.add_argument('--alpha',
                                   default=0.5,
                                   type=float,
                                   help='float between 0 and 1')
    elif method == LateFusionMethod.mean_and_min:
        parser_method.add_argument('--weights', nargs='+', default=[], type=float, help='Weights for the mean part')
        parser_method.add_argument('--gamma',
                                   default=0.5,
                                   type=float,
                                   help=('float between 0 and 1; weight between mean and min:'
                                         'gamma * mean + (1-gamma) * min'))
    elif method == LateFusionMethod.mean_and_power:
        parser_method.add_argument('--weights', nargs='+', default=[], type=float,
                                   help='Weights for the mean/power part')
        parser_method.add_argument('--gamma',
                                   default=0.5,
                                   type=float,
                                   help=('float between 0 and 1; weight between mean and power:'
                                         'gamma * mean + (1-gamma) * power'))
    elif method == LateFusionMethod.generalized_harmonic_mean:
        parser_method.add_argument('--weights', nargs='+', default=[], type=float, help='Weights for the mean part')
        parser_method.add_argument('--gamma',
                                   default=0.5,
                                   type=float,
                                   help='float >= 0')
    return parser_method
