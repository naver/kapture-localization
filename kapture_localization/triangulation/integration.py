# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import numpy as np
import quaternion
import cv2
import scipy.special
import itertools
from collections import defaultdict
from functools import partial

from kapture_localization.utils.logging import getLogger
from kapture_localization.triangulation.triangulate import triangulate_n_views_ransac
from kapture_localization.triangulation.triangulate import get_inliers
from kapture_localization.utils.cv_camera_matrix import get_camera_matrix_from_kapture

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
from kapture.io.features import image_keypoints_from_file
from kapture.io.features import get_matches_fullpath, image_matches_from_file


def aggregate_matches(image_name, pairs, map_name_to_tuple_key, keypoints_filepaths,
                      kapture_data, keypoints_type, kapture_path, tar_handlers):
    """
    gather kpid -> matches keypoints for all map images
    """
    # gather kpid -> matches keypoints for all map images
    aggregated_matches = defaultdict(list)
    for img_map in pairs:
        map_ts, map_sensor_id = map_name_to_tuple_key[img_map]
        map_keypoints_filepath = keypoints_filepaths[img_map]
        kapture_keypoints_map = image_keypoints_from_file(filepath=map_keypoints_filepath,
                                                          dsize=kapture_data.keypoints[keypoints_type].dsize,
                                                          dtype=kapture_data.keypoints[keypoints_type].dtype)
        map_cam = kapture_data.sensors[map_sensor_id]
        assert isinstance(map_cam, kapture.Camera)
        map_num_keypoints = kapture_keypoints_map.shape[0]
        kapture_keypoints_map, KMap, distortionMap = get_camera_matrix_from_kapture(kapture_keypoints_map, map_cam)
        kapture_keypoints_map = kapture_keypoints_map.reshape((map_num_keypoints, 2))

        pose_map = kapture_data.trajectories[map_ts, map_sensor_id]
        rotation = quaternion.as_rotation_matrix(pose_map.r)
        translation = pose_map.t
        extrinsic = np.column_stack((rotation, translation))
        projection_matrix = np.dot(KMap, extrinsic)

        # get matches
        if image_name < img_map:
            if (image_name, img_map) not in kapture_data.matches[keypoints_type]:
                getLogger().warning(f'pair {image_name}, {img_map} do not have a match file, skipped')
                continue
            matches_path = get_matches_fullpath((image_name, img_map), keypoints_type, kapture_path, tar_handlers)
        else:
            if (img_map, image_name) not in kapture_data.matches[keypoints_type]:
                getLogger().warning(f'pair {image_name}, {img_map} do not have a match file, skipped')
            matches_path = get_matches_fullpath((img_map, image_name), keypoints_type, kapture_path, tar_handlers)
        matches = image_matches_from_file(matches_path)

        if image_name < img_map:
            i_query = 0
            i_map = 1
        else:
            i_query = 1
            i_map = 0
        kpt_id_matches = matches[:, 0:2].astype(int)
        useful_kpts_map = kapture_keypoints_map[kpt_id_matches[:, i_map]]
        if useful_kpts_map.shape[0] > 0 and np.count_nonzero(distortionMap) > 0:
            epsilon = np.finfo(np.float64).eps
            stop_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, epsilon)
            useful_kpts_map = cv2.undistortPointsIter(useful_kpts_map, KMap, distortionMap,
                                                      R=None, P=KMap,
                                                      criteria=stop_criteria)
        useful_kpts_map = useful_kpts_map.reshape((kpt_id_matches.shape[0], 2))

        def match_to_tuple(x, y): return (projection_matrix, useful_kpts_map[x], img_map, y)
        for ind, m in enumerate(kpt_id_matches):
            aggregated_matches[m[i_query].item()].append(match_to_tuple(ind, m[i_map].item()))

    return aggregated_matches


def _triangulate_one_point(sample_count, max_num_iterations,
                           min_required_inliers, inlier_threshold,
                           map_points_tuple):
    pid, map_points = map_points_tuple
    if len(map_points) < sample_count:
        # not enough to triangulate an observation
        return False, pid, None, None
    number_of_views = len(map_points)
    np_views = np.array([kpt for _, kpt, _, _ in map_points], dtype=np.float64)
    np_projection_matrices = np.array([projmat for projmat, _, _, _ in map_points], dtype=np.float64)

    num_permutation = int(scipy.special.binom(number_of_views, sample_count))
    num_iteration = min(max_num_iterations, num_permutation)
    if num_iteration == num_permutation:
        indexes = np.arange(number_of_views)
        combinations = list(itertools.combinations(indexes, r=sample_count))
    else:
        combinations = []

    success, point3d, num_inliers, residuals = triangulate_n_views_ransac(
        np_views,
        np_projection_matrices,
        min_required_inliers,
        sample_count,
        inlier_threshold,
        num_iteration,
        np.array(combinations, dtype=np.int64, ndmin=2))

    if success:
        inliers = get_inliers(np.int64(num_inliers), residuals, inlier_threshold)
        inliers_ref = []
        for i in inliers:
            map_name, map_kpid = map_points[i][2], map_points[i][3]
            inliers_ref.append((map_name, map_kpid))
        return True, pid, point3d, inliers_ref

    return False, pid, None, None


def triangulate_all_points(pool,
                           aggregated_matches,
                           max_num_iterations,
                           inlier_threshold,
                           keypoints_type):
    """
    triangulate all points from a kpid -> matches dict obtained with aggregate_matches
    """
    pid = 0
    sample_count = 3
    min_required_inliers = 3

    result_observations = kapture.Observations()
    result_points3d = []
    point_id_from_obs = {}
    kpid_to_point3d = {}

    triangulate_one_point_fun = partial(_triangulate_one_point,
                                        sample_count, max_num_iterations,
                                        min_required_inliers, inlier_threshold)
    # debug lines
    # results_pool = []
    # for v in aggregated_matches:
    #     results_pool.append(triangulate_one_point_fun(v))
    aggregated_matches_short = {k: v for k, v in aggregated_matches.items() if len(v) >= sample_count}
    results_pool = pool.map(triangulate_one_point_fun, aggregated_matches_short.items())

    for success, kpid, point3d, inliers in results_pool:
        if success:
            for map_name, map_kpid in inliers:
                result_observations.add(pid, keypoints_type, map_name, map_kpid)
                point_id_from_obs[(map_name, map_kpid)] = pid
            result_points3d.append(point3d)
            kpid_to_point3d[kpid] = point3d
            pid += 1

    result_points3d_np = np.zeros((len(result_points3d), 6))
    if len(result_points3d) > 0:
        result_points3d_np[:, 0:3] = np.array(result_points3d)
    return point_id_from_obs, result_observations, kapture.Points3d(result_points3d_np), kpid_to_point3d


def convert_correspondences(kpid_to_point3d,
                            kpts_query,
                            kpts_query_undistorted):
    """
    convert kpid_to_point3d obtained from triangulate_all_points
    to tuple(points2D, points2D_undistorted, points3D, stats) usable with pycolmap/pyransaclib
    """
    points2D = []
    points2D_undistorted = []
    points3D = []
    for kpid_query, point3d in kpid_to_point3d.items():
        if kpts_query is not None:
            kp_query = kpts_query[kpid_query]
            points2D.append(kp_query[0:2])
        if kpts_query_undistorted is not None:
            kp_query_undistorted = kpts_query_undistorted[kpid_query]
            points2D_undistorted.append(kp_query_undistorted[0:2])
        points3D.append(point3d)
    stats = {}

    return points2D, points2D_undistorted, points3D, stats
