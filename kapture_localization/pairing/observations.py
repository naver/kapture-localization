# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import logging
from itertools import combinations
import multiprocessing
from typing import Dict, List, Optional, Tuple, Set
import gc
from tqdm import tqdm

from kapture_localization.utils.logging import getLogger

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture


def _child_process_get_pairs(kdata_observations: List[Tuple[str, int]],
                             imgs_map: Set[str],
                             imgs_query: Optional[Set[str]]):
    result_pairs = {}
    pairs = list(combinations(kdata_observations, r=2))  # get all pairs from 3D point observations
    if len(pairs) > 1:
        for p in pairs:
            img1 = p[0][0]
            img2 = p[1][0]
            if img1 == img2:
                # skip pair if both images are the same
                continue
            if imgs_query is not None:
                # if query images are different from mapping images, i.e. if query kapture was provided
                if img1 in imgs_map and img1 not in imgs_query and img2 in imgs_map and img2 not in imgs_query:
                    # skip if both images are from the mapping kapture
                    # (because we only want to keep query-mapping pairs)
                    continue
                if img1 in imgs_query and img2 in imgs_query:
                    # skip if both images are from the query kapture (because we do not want to match query-query)
                    continue
                # ensure query-mapping order in the pair file
                if img1 in imgs_query:
                    pair = (img1, img2)
                else:
                    pair = (img2, img1)
                if not pair[0] in result_pairs:
                    result_pairs[pair[0]] = {}
                if not pair[1] in result_pairs[pair[0]]:
                    result_pairs[pair[0]][pair[1]] = 0
                result_pairs[pair[0]][pair[1]] += 1
            else:
                if img1 not in imgs_map or img2 not in imgs_map:
                    continue

                # ensure lexicographic order of the pairs
                if img1 < img2:
                    pair = (img1, img2)
                else:
                    pair = (img2, img1)
                if not pair[0] in result_pairs:
                    result_pairs[pair[0]] = {}
                if not pair[1] in result_pairs[pair[0]]:
                    result_pairs[pair[0]][pair[1]] = 0
                result_pairs[pair[0]][pair[1]] += 1
    return result_pairs


def get_observation_image_pairs(keypoints_type: str,
                                kdata: kapture.Kapture,
                                kdata_query: Optional[kapture.Kapture],
                                max_number_of_threads: Optional[int] = None):
    """
    get observations pairs as dictionary
    """
    assert kdata.records_camera is not None
    imgs_map = kdata.records_camera.data_list()
    if kdata_query is not None:
        assert kdata_query.records_camera is not None
        imgs_query = kdata_query.records_camera.data_list()
    else:
        imgs_query = None
    all_pairs = {}

    number_of_threads = multiprocessing.cpu_count() if max_number_of_threads is None else max_number_of_threads

    def update_all_pairs_and_progress_bar(result):
        for img1 in result:
            if img1 not in all_pairs:
                all_pairs[img1] = {}
            for img2 in result[img1]:
                if img2 not in all_pairs[img1]:
                    all_pairs[img1][img2] = 0
                all_pairs[img1][img2] += result[img1][img2]
        progress_bar.update(1)

    def error_callback(e):
        getLogger().critical(e)

    getLogger().debug(f'computing all possible pairs from observations, max-threads={number_of_threads}')
    assert kdata.observations is not None
    progress_bar = tqdm(total=len(kdata.observations),
                        disable=getLogger().level >= logging.CRITICAL)

    imgs_map_set = set(imgs_map)
    imgs_query_set = set(imgs_query) if imgs_query is not None else None
    with multiprocessing.Pool(number_of_threads) as pool:
        for point3d_id in kdata.observations.keys():
            if keypoints_type not in kdata.observations[point3d_id]:
                progress_bar.update(1)
                continue
            pool.apply_async(_child_process_get_pairs, args=(kdata.observations[point3d_id, keypoints_type],
                                                             imgs_map_set,
                                                             imgs_query_set),
                             callback=update_all_pairs_and_progress_bar,
                             error_callback=error_callback)
        pool.close()
        pool.join()
    progress_bar.close()
    return all_pairs


def _child_process_get_observation_images(kdata_observations: List[Tuple[str, int]],
                                          imgs_map: Set[str],
                                          imgs_query: Optional[Set[str]]):
    result_observations = {}
    for image_name, _ in kdata_observations:
        if image_name not in imgs_map and (imgs_query is None or image_name not in imgs_query):
            continue
        if image_name not in result_observations:
            result_observations[image_name] = 0
        result_observations[image_name] += 1
    return result_observations


def get_observation_images(keypoints_type: str,
                           kdata: kapture.Kapture,
                           kdata_query: Optional[kapture.Kapture],
                           max_number_of_threads: Optional[int] = None):
    """
    get a dictionary image -> number of observations
    """
    assert kdata.records_camera is not None
    imgs_map = kdata.records_camera.data_list()
    if kdata_query is not None:
        assert kdata_query.records_camera is not None
        imgs_query = kdata_query.records_camera.data_list()
    else:
        imgs_query = None

    result_observations = {}
    number_of_threads = multiprocessing.cpu_count() if max_number_of_threads is None else max_number_of_threads

    def update_observations_and_progress_bar(result):
        for img1, count in result.items():
            if img1 not in result_observations:
                result_observations[img1] = 0
            result_observations[img1] += count
        progress_bar.update(1)

    def error_callback(e):
        getLogger().critical(e)

    getLogger().debug(f'computing all possible pairs from observations, max-threads={number_of_threads}')
    assert kdata.observations is not None
    progress_bar = tqdm(total=len(kdata.observations),
                        disable=getLogger().level >= logging.CRITICAL)

    imgs_map_set = set(imgs_map)
    imgs_query_set = set(imgs_query) if imgs_query is not None else None
    with multiprocessing.Pool(number_of_threads) as pool:
        for point3d_id in kdata.observations.keys():
            if keypoints_type not in kdata.observations[point3d_id]:
                progress_bar.update(1)
                continue
            pool.apply_async(_child_process_get_observation_images,
                             args=(kdata.observations[point3d_id, keypoints_type],
                                   imgs_map_set,
                                   imgs_query_set),
                             callback=update_observations_and_progress_bar,
                             error_callback=error_callback)
        pool.close()
        pool.join()
    progress_bar.close()
    return result_observations


def get_topk_observation_pairs(all_pairs: Dict[str, Dict[str, int]],
                               records_camera: kapture.RecordsCamera,
                               topk: int):
    """
    convert pairs dict to list
    """
    image_pairs = []
    for img1 in sorted(records_camera.data_list()):
        if img1 not in all_pairs:
            getLogger().debug(f'{img1} has no images sharing observations')
            continue
        sorted_pairs = list(sorted(all_pairs[img1].items(), key=lambda item: item[1], reverse=True))
        for img2, score in sorted_pairs[0:topk]:
            image_pairs.append((img1, img2, score))
    return image_pairs


def get_pairs_observations(kdata: kapture.Kapture,
                           kdata_query: Optional[kapture.Kapture],
                           keypoints_type: str,
                           max_number_of_threads: Optional[int],
                           iou: bool,
                           topk: int):
    """
    get observations pairs as list
    """
    if iou:
        individual_observations = get_observation_images(keypoints_type,
                                                         kdata, kdata_query,
                                                         max_number_of_threads)
        gc.collect()
    else:
        individual_observations = None
    all_pairs = get_observation_image_pairs(keypoints_type,
                                            kdata, kdata_query,
                                            max_number_of_threads)
    if iou:
        assert individual_observations is not None
        final_pairs = {}
        for img1 in all_pairs.keys():
            for img2 in all_pairs[img1].keys():
                if img1 not in final_pairs:
                    final_pairs[img1] = {}
                union = individual_observations[img1] + individual_observations[img2] - all_pairs[img1][img2]
                if union == 0:
                    final_pairs[img1][img2] = 0
                else:
                    final_pairs[img1][img2] = all_pairs[img1][img2] / union
        all_pairs = final_pairs

    getLogger().info('ranking co-observation pairs...')
    assert kdata.records_camera is not None
    image_pairs = get_topk_observation_pairs(all_pairs, kdata.records_camera, topk)
    return image_pairs
