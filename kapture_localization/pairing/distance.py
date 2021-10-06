# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import logging
import multiprocessing
from typing import Optional
from tqdm import tqdm

from kapture_localization.utils.logging import getLogger

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
from kapture.algo.pose_operations import world_pose_transform_distance


def _child_process_get_pairs(img_query: str, pose_query: kapture.PoseTransform,
                             img_map: str, pose_map: kapture.PoseTransform,
                             min_distance: float,
                             max_distance: float,
                             max_angle: float):
    distance, rotation_distance = world_pose_transform_distance(pose_map, pose_query)
    is_rejected = (distance < min_distance or distance > max_distance or rotation_distance > max_angle)
    score = 2.0 - ((min(distance, max_distance)/max_distance) + (min(rotation_distance, max_angle) / max_angle))
    return not is_rejected, (img_query, img_map, score)


def get_image_pairs_from_distance(kdata: kapture.Kapture,
                                  kdata_query: Optional[kapture.Kapture],
                                  min_distance: float,
                                  max_distance: float,
                                  max_angle: float,
                                  keep_rejected: bool,
                                  max_number_of_threads: Optional[int] = None):
    """
    get pairs as dictionary from distance
    """
    if kdata.rigs is None:
        map_trajectories = kdata.trajectories
    else:
        map_trajectories = kapture.rigs_remove(kdata.trajectories, kdata.rigs)

    imgs_map = [(img, map_trajectories[ts, sensor_id])
                for ts, sensor_id, img in kapture.flatten(kdata.records_camera)
                if (ts, sensor_id) in map_trajectories]
    if kdata_query is not None:
        if kdata_query.rigs is None:
            query_trajectories = kdata_query.trajectories
        else:
            query_trajectories = kapture.rigs_remove(kdata_query.trajectories, kdata_query.rigs)
        imgs_query = [(img, query_trajectories[ts, sensor_id])
                      for ts, sensor_id, img in kapture.flatten(kdata_query.records_camera)
                      if (ts, sensor_id) in query_trajectories]
        # create all possible pairs
        filepath_pairs = [(f1, pose_f1, f2, pose_f2)
                          for f1, pose_f1 in imgs_query
                          for f2, pose_f2 in imgs_map
                          if f1 != f2]
    else:
        imgs_query = imgs_map

        # create all possible pairs
        filepath_pairs = [(f1, pose_f1, f2, pose_f2)
                          for f1, pose_f1 in imgs_map
                          for f2, pose_f2 in imgs_map
                          if f1 < f2]

    all_pairs = {img: [] for img, _ in imgs_query}
    number_of_threads = multiprocessing.cpu_count() if max_number_of_threads is None else max_number_of_threads

    def _update_all_pairs_and_progress_bar(result):
        if result[0] or keep_rejected:
            all_pairs[result[1][0]].append((result[1][1], result[1][2]))
        progress_bar.update(1)

    def _error_callback(e):
        getLogger().critical(e)

    getLogger().debug(f'computing all possible pairs from distance, max-threads={number_of_threads}')
    progress_bar = tqdm(total=len(filepath_pairs),
                        disable=getLogger().level >= logging.CRITICAL)

    # for f1, pose_f1, f2, pose_f2 in filepath_pairs:
    #     result = _child_process_get_pairs(f1, pose_f1,
    #                                       f2, pose_f2,
    #                                       min_distance,
    #                                       max_distance,
    #                                       max_angle)
    #     _update_all_pairs_and_progress_bar(result)

    with multiprocessing.Pool(number_of_threads) as pool:
        for f1, pose_f1, f2, pose_f2 in filepath_pairs:
            pool.apply_async(_child_process_get_pairs, args=(f1, pose_f1,
                                                             f2, pose_f2,
                                                             min_distance,
                                                             max_distance,
                                                             max_angle),
                             callback=_update_all_pairs_and_progress_bar,
                             error_callback=_error_callback)
        pool.close()
        pool.join()
    progress_bar.close()
    return all_pairs


def get_pairs_distance(kdata: kapture.Kapture,
                       kdata_query: Optional[kapture.Kapture],
                       topk: Optional[int],
                       min_distance: float,
                       max_distance: float,
                       max_angle: float,
                       keep_rejected: bool,
                       max_number_of_threads: Optional[int] = None):
    """
    get pairs as list from distance
    """
    all_pairs = get_image_pairs_from_distance(kdata, kdata_query,
                                              min_distance,
                                              max_distance,
                                              max_angle,
                                              keep_rejected,
                                              max_number_of_threads)
    if kdata_query is not None:
        image_pairs = []
        for query_image_name, images_to_match in sorted(all_pairs.items()):
            k = 0
            for mapping_image_name, score in sorted(images_to_match, key=lambda x: x[1], reverse=True):
                if topk is not None and k >= topk:
                    break
                # don't match image with itself
                if query_image_name == mapping_image_name:
                    continue
                image_pairs.append([query_image_name, mapping_image_name, score])
                k += 1
    else:
        # put back the duplicates (to make the pairfile easy to read)
        dup_pairs = {img: [] for img in all_pairs.keys()}
        for img1, paired_images in all_pairs.items():
            for img2, score in paired_images:
                dup_pairs[img2].append((img1, score))
        for img in all_pairs.keys():
            all_pairs[img].extend(dup_pairs[img])
            del dup_pairs[img]
        image_pairs = []
        for query_image_name, images_to_match in sorted(all_pairs.items()):
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
