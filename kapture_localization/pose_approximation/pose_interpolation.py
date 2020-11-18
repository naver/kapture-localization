# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from typing import Dict, Tuple, List
import kapture
from kapture.algo.pose_operations import average_pose_transform_weighted


def get_interpolated_pose(kdata_map: kapture.Kapture, kdata_query: kapture.Kapture,
                          weights: Dict[str, List[Tuple[str, float]]]):
    """
    compute the approximated pose for all query images given the precomputed weights

    :param kdata_map: map images + their poses as kapture data
    :type kdata_map: kapture.Kapture
    :param kdata_query: query images as kapture data
    :type kdata_query: kapture.Kapture
    :param weights: weights for the pose interpolation
    :type weights: Dict[str, List[Tuple[str, float]]]
    """
    output_trajectories = kapture.Trajectories()
    assert kdata_map.trajectories is not None
    assert kdata_map.records_camera is not None
    reverse_map_records_camera = {image_name: (timestamp, sensor_id)
                                  for timestamp, sensor_id, image_name in kapture.flatten(kdata_map.records_camera)}
    if kdata_map.rigs is not None:
        input_trajectories = kapture.rigs_remove(kdata_map.trajectories, kdata_map.rigs)
    else:
        input_trajectories = kdata_map.trajectories

    assert kdata_query.records_camera is not None
    reverse_query_records_camera = {image_name: (timestamp, sensor_id)
                                    for timestamp, sensor_id, image_name in kapture.flatten(kdata_query.records_camera)}

    for query_image_name, weighted_map_images in weights.items():
        pose_inv_list = [input_trajectories[reverse_map_records_camera[name]].inverse()
                         for name, _ in weighted_map_images]
        weight_list = [w for _, w in weighted_map_images]
        output_trajectories[reverse_query_records_camera[query_image_name]] = average_pose_transform_weighted(
            pose_inv_list,
            weight_list
        ).inverse()
    return output_trajectories
