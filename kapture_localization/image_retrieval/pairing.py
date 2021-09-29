# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import logging
from typing import List, Tuple, Union, Dict, Optional
import numpy as np
from tqdm import tqdm

from kapture_localization.utils.logging import getLogger

import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.io.csv import GlobalFeaturesConfig
from kapture.io.features import image_global_features_from_file
from kapture.io.tar import TarHandler


class StackedGlobalFeatures:
    def __init__(self,
                 stacked_features_index: np.ndarray,
                 stacked_features: np.ndarray) -> None:
        self.index = stacked_features_index
        self.stacked_features = stacked_features


def stack_global_features(global_features_config: GlobalFeaturesConfig,
                          global_features_paths: List[Tuple[str, Union[str, Tuple[str, TarHandler]]]]
                          ) -> Tuple[Union[np.array, List[str]], np.ndarray]:
    """
    loads global features and store them inside a numpy array of shape (number_of_images, dsize)

    :param global_features_config: content of global_features.txt, required to load the global features
    :type global_features_config: GlobalFeaturesConfig
    :param global_features_paths: list of every image and the full path to the corresponding global feature
    :type global_features_paths: List[Tuple[str, str]]
    """
    getLogger().debug('loading global features')
    number_of_images = len(global_features_paths)

    stacked_features_index = np.empty((number_of_images,), dtype=object)
    stacked_features = np.empty((number_of_images, global_features_config.dsize),
                                dtype=global_features_config.dtype)

    hide_progress_bar = getLogger().getEffectiveLevel() > logging.INFO
    for i, (image_path, global_feature_path) in tqdm(enumerate(global_features_paths), disable=hide_progress_bar):
        stacked_features_index[i] = image_path
        global_feature = image_global_features_from_file(global_feature_path,
                                                         global_features_config.dtype,
                                                         global_features_config.dsize)
        global_feature = global_feature / np.linalg.norm(global_feature)
        stacked_features[i] = global_feature

    return StackedGlobalFeatures(stacked_features_index, stacked_features)


def get_similarity_matrix(query_features: StackedGlobalFeatures, map_features: StackedGlobalFeatures):
    return query_features.stacked_features.dot(map_features.stacked_features.T)


def get_similarity(query_features: StackedGlobalFeatures,
                   map_features: StackedGlobalFeatures) -> Dict[str, List[Tuple[str, float]]]:
    """
    get similarity in the form of a dictionary

    :param query_features: stacked query global features
    :type query_features: StackedGlobalFeatures
    :param map_features: stacked map global features
    :type map_features: StackedGlobalFeatures
    :return: query_name -> sorted (high score first) list [(mapping_name, score), ...]
    :rtype: Dict[str, List[Tuple[str, float]]]
    """
    similarity_matrix = get_similarity_matrix(query_features, map_features)
    return get_similarity_dict_from_similarity_matrix(similarity_matrix, query_features.index, map_features.index)


def get_similarity_dict_from_similarity_matrix(
    similarity_matrix: np.ndarray,
    query_features_index: Union[np.array, List[str]],
    map_features_index: Union[np.array, List[str]],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    convert similarity_matrix to a dictionary
    """
    similarity_dict = {}
    for i, line in enumerate(similarity_matrix):
        scores = line
        indexes = np.argsort(-scores)
        query_name = query_features_index[i]
        similarity_dict[query_name] = list(zip(map_features_index[indexes], scores[indexes]))
    return similarity_dict


def get_image_pairs(similarity: Dict[str, List[Tuple[str, float]]],
                    topk: Optional[int] = None) -> List[Tuple[str, str, float]]:
    """
    convert similarity dictionary to list of pairs

    :param similarity: result of get_similarity()
    :type similarity: Dict[str, List[Tuple[str, float]]]
    :param topk: number of images retained for each query, defaults to None
    :type topk: Optional[int], optional
    :return: list(image_name_query, image_name_map, score)
    :rtype: List[Tuple[str, str, float]]
    """
    image_pairs = []
    for query_image_name, images_to_match in sorted(similarity.items()):
        k = 0
        for mapping_image_name, score in sorted(images_to_match, key=lambda x: x[1], reverse=True):
            if topk is not None and k >= topk:
                break
            # don't match image with itself
            if query_image_name == mapping_image_name:
                continue
            image_pairs.append([query_image_name, mapping_image_name, score])
            k += 1
    return image_pairs
