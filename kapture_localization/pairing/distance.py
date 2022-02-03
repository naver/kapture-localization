# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
from typing import Optional
import numpy as np
import quaternion
import math

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture


def get_position_diff(imgs_query, imgs_map):
    positions_query = np.empty((len(imgs_query), 3, 1), dtype=np.float32)
    for i, (_, pose) in enumerate(imgs_query):
        positions_query[i, :, 0] = pose.t_raw
    positions_query_tile = np.tile(positions_query, (1, 1, len(imgs_map)))

    positions_map = np.empty((1, 3, len(imgs_map)), dtype=np.float32)
    for i, (_, pose) in enumerate(imgs_map):
        positions_map[0, :, i] = pose.t_raw
    positions_map_tile = np.tile(positions_map, (len(imgs_query), 1, 1))
    return np.sqrt(np.sum(np.square(positions_query_tile - positions_map_tile), axis=1))


def get_rotations_diff(imgs_query, imgs_map):
    rad_to_deg = 180.0 / math.pi

    rotations_query = np.empty((len(imgs_query), 1), dtype=np.quaternion)
    for i, (_, pose) in enumerate(imgs_query):
        rotations_query[i, 0] = pose.r
    rotations_query_tile = np.tile(rotations_query, (1, len(imgs_map)))

    rotations_map = np.empty((1, len(imgs_map)), dtype=np.quaternion)
    for i, (_, pose) in enumerate(imgs_map):
        rotations_map[0, i] = pose.r
    rotations_map_tile = np.tile(rotations_map, (len(imgs_query), 1))
    return quaternion.rotation_intrinsic_distance(rotations_query_tile, rotations_map_tile) * rad_to_deg


def get_pairs_distance(kdata: kapture.Kapture,
                       kdata_query: kapture.Kapture,
                       topk: Optional[int],
                       min_distance: float,
                       max_distance: float,
                       max_angle: float,
                       keep_rejected: bool):
    """
    get pairs as list from distance
    """
    if kdata.rigs is None:
        map_trajectories = kdata.trajectories
    else:
        map_trajectories = kapture.rigs_remove(kdata.trajectories, kdata.rigs)

    imgs_map = [(img, map_trajectories[ts, sensor_id].inverse())
                for ts, sensor_id, img in kapture.flatten(kdata.records_camera)
                if (ts, sensor_id) in map_trajectories]

    if kdata_query.rigs is None:
        query_trajectories = kdata_query.trajectories
    else:
        query_trajectories = kapture.rigs_remove(kdata_query.trajectories, kdata_query.rigs)
    imgs_query = [(img, query_trajectories[ts, sensor_id].inverse())
                  for ts, sensor_id, img in kapture.flatten(kdata_query.records_camera)
                  if (ts, sensor_id) in query_trajectories]

    positions_scores = get_position_diff(imgs_query, imgs_map)
    rotation_scores = get_rotations_diff(imgs_query, imgs_map)

    # is_rejected = (distance < min_distance or distance > max_distance or rotation_distance > max_angle)
    ones = np.ones(positions_scores.shape)

    invalid = (positions_scores < (ones * min_distance)) | \
        (positions_scores > (ones * max_distance)) | \
        (rotation_scores > (ones * max_angle))

    score = (ones * 2.0) - \
        (np.minimum(positions_scores, ones * max_distance)/max_distance
         +
         np.minimum(rotation_scores, ones * max_angle)/max_angle)

    similarity_dict = {}
    for i, line in enumerate(score):
        scores = line
        indexes = np.argsort(-scores)
        query_name = imgs_query[i][0]
        pairs = []
        k = 0
        for j in indexes:
            if topk is not None and k >= topk:
                break
            if not keep_rejected and invalid[i, j]:
                continue
            map_name = imgs_map[j][0]
            if query_name == map_name:
                continue
            pairs.append((map_name, scores[j]))
            k += 1
        similarity_dict[query_name] = pairs

    image_pairs = []
    for query_image_name, images_to_match in sorted(similarity_dict.items()):
        for mapping_image_name, score in images_to_match:
            image_pairs.append([query_image_name, mapping_image_name, score])
    return image_pairs
