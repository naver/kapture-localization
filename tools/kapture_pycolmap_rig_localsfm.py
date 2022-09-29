#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script localize images from a multi camera rig using pycolmap from kapture format
"""

import argparse
import logging
import os
from tqdm import tqdm
from typing import Optional, List
try:
    import pycolmap
    has_pycolmap = True
except ModuleNotFoundError:
    has_pycolmap = False

import datetime
import numpy as np
import multiprocessing

import path_to_kapture_localization  # noqa: F401
from kapture_localization.utils.logging import getLogger, save_to_json
from kapture_localization.localization.reprojection_error import compute_reprojection_error
from kapture_localization.utils.cv_camera_matrix import get_camera_matrix_from_kapture
from kapture_localization.utils.rigs_extension import get_top_level_rig_ids, get_all_cameras_from_rig_ids
from kapture_localization.triangulation.integration import aggregate_matches, triangulate_all_points
from kapture_localization.triangulation.integration import convert_correspondences

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.io.csv
import kapture.utils.logging
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.io.features import keypoints_to_filepaths, image_keypoints_from_file
from kapture.io.structure import delete_existing_kapture_files
from kapture.converter.colmap.cameras import get_colmap_camera, CAMERA_MODEL_NAMES
from kapture.utils.Collections import try_get_only_key_from_collection
from kapture.io.tar import TarCollection


logger = logging.getLogger('kapture_pycolmap_rig_localsfm')


def kapture_pycolmap_rig_localsfm(kapture_path: str,
                                  kapture_query_path: str,
                                  output_path: str,
                                  pairsfile_path: str,
                                  rig_ids: List[str],
                                  apply_rigs_remove: bool,
                                  max_error: float,
                                  min_inlier_ratio: float,
                                  min_num_iterations: int,
                                  max_num_iterations: int,
                                  confidence: float,
                                  keypoints_type: Optional[str],
                                  write_detailed_report: bool,
                                  max_number_of_threads: Optional[int],
                                  force: bool) -> None:
    """
    Localize images from a multi camera rig using pycolmap

    :param kapture_path: path to the kapture to use (mapping and query images)
    :param kapture_query_path: path to the kapture to use (query images)
    :param output_path: path to the write the localization results
    :param pairsfile_path: pairs to use
    :param rig_ids: list of rig ids that should be localized
    :param apply_rigs_remove: apply rigs remove before saving poses to disk
    :param max_error: RANSAC inlier threshold in pixel, shared between all cameras
    :param min_inlier_ratio: abs_pose_options.ransac_options.min_inlier_ratio
    :param min_num_iterations: abs_pose_options.ransac_options.min_num_trials
    :param max_num_iterations: abs_pose_options.ransac_options.max_num_trials
    :param confidence: abs_pose_options.ransac_options.confidence
    :param keypoints_type: types of keypoints (and observations) to use
    :param duplicate_strategy: strategy to handle duplicate correspondences (either kpt_id and/or pt3d_id)
    :param rerank_strategy: strategy to reorder pairs before handling duplicate correspondences
    :param write_detailed_report: if True, write a json file with inliers, reprojection error for each query
    :param max_number_of_threads: maximum number of parallel triangulations running
    :param force: Silently overwrite kapture files if already exists.
    """
    # Load input files first to make sure it is OK
    logger.info('loading kapture files...')
    with kapture.io.csv.get_all_tar_handlers(kapture_path) as tar_handlers:
        kapture_data = kapture.io.csv.kapture_from_dir(kapture_path,
                                                       pairsfile_path,
                                                       [kapture.GlobalFeatures,
                                                        kapture.Descriptors,
                                                        kapture.Points3d,
                                                        kapture.Observations],
                                                       tar_handlers)
        kapture_query_data = kapture.io.csv.kapture_from_dir(kapture_query_path,
                                                             None,
                                                             [kapture.Keypoints,
                                                              kapture.Descriptors,
                                                              kapture.GlobalFeatures,
                                                              kapture.Matches,
                                                              kapture.Points3d,
                                                              kapture.Observations])

        kapture_pycolmap_rig_localsfm_from_loaded_data(kapture_data,
                                                       kapture_path,
                                                       tar_handlers,
                                                       kapture_query_data,
                                                       output_path,
                                                       pairsfile_path,
                                                       rig_ids,
                                                       apply_rigs_remove,
                                                       max_error,
                                                       min_inlier_ratio,
                                                       min_num_iterations,
                                                       max_num_iterations,
                                                       confidence,
                                                       keypoints_type,
                                                       write_detailed_report,
                                                       max_number_of_threads,
                                                       force)


def kapture_pycolmap_rig_localsfm_from_loaded_data(
        kapture_data: kapture.Kapture,
        kapture_path: str,
        tar_handlers: TarCollection,
        kapture_query_data: kapture.Kapture,
        output_path: str,
        pairsfile_path: str,
        rig_ids: List[str],
        apply_rigs_remove: bool,
        max_error: float,
        min_inlier_ratio: float,
        min_num_iterations: int,
        max_num_iterations: int,
        confidence: float,
        keypoints_type: Optional[str],
        write_detailed_report: bool,
        max_number_of_threads: Optional[int],
        force: bool) -> None:
    """
    Localize images from a multi camera rig using pycolmap

    :param kapture_data: loaded kapture data (mapping and query images, incl. matches)
    :param kapture_path: path to the kapture to use
    :param tar_handlers: collection of pre-opened tar archives
    :param kapture_query_data: loaded kapture data (query images)
    :param output_path: path to the write the localization results
    :param pairsfile_path: pairs to use
    :param rig_ids: list of rig ids that should be localized
    :param apply_rigs_remove: apply rigs remove before saving poses to disk
    :param max_error: RANSAC inlier threshold in pixel, shared between all cameras
    :param min_inlier_ratio: abs_pose_options.ransac_options.min_inlier_ratio
    :param min_num_iterations: abs_pose_options.ransac_options.min_num_trials
    :param max_num_iterations: abs_pose_options.ransac_options.max_num_trials
    :param confidence: abs_pose_options.ransac_options.confidence
    :param keypoints_type: types of keypoints (and observations) to use
    :param duplicate_strategy: strategy to handle duplicate correspondences (either kpt_id and/or pt3d_id)
    :param rerank_strategy: strategy to reorder pairs before handling duplicate correspondences
    :param write_detailed_report: if True, write a json file with inliers, reprojection error for each query
    :param max_number_of_threads: maximum number of parallel triangulations running
    :param force: Silently overwrite kapture files if already exists.
    """
    assert has_pycolmap
    if not (kapture_data.records_camera and kapture_data.sensors and kapture_data.keypoints and
            kapture_data.matches):
        raise ValueError('records_camera, sensors, keypoints, matches are mandatory for map+query')

    if not (kapture_query_data.records_camera and kapture_query_data.sensors):
        raise ValueError('records_camera, sensors are mandatory for query')

    if keypoints_type is None:
        keypoints_type = try_get_only_key_from_collection(kapture_data.keypoints)
    assert keypoints_type is not None
    assert keypoints_type in kapture_data.keypoints
    assert keypoints_type in kapture_data.matches

    assert kapture_query_data.rigs is not None
    assert len(kapture_query_data.rigs) >= 1
    if len(rig_ids) == 0:
        rig_ids = get_top_level_rig_ids(kapture_query_data.rigs)
    final_camera_list = get_all_cameras_from_rig_ids(rig_ids, kapture_query_data.sensors, kapture_query_data.rigs)
    assert len(final_camera_list) > 0

    if kapture_query_data.trajectories is not None:
        logger.warning("Input query data contains trajectories: they will be ignored")
        kapture_query_data.trajectories.clear()

    os.umask(0o002)
    os.makedirs(output_path, exist_ok=True)
    delete_existing_kapture_files(output_path, force_erase=force)

    # load pairsfile
    pairs = {}
    with open(pairsfile_path, 'r') as fid:
        table = kapture.io.csv.table_from_file(fid)
        for img_query, img_map, _ in table:
            if img_query not in pairs:
                pairs[img_query] = []
            pairs[img_query].append(img_map)

    kapture_data.matches[keypoints_type].normalize()
    keypoints_filepaths = keypoints_to_filepaths(kapture_data.keypoints[keypoints_type],
                                                 keypoints_type,
                                                 kapture_path,
                                                 tar_handlers)

    timestamps = list(kapture_query_data.records_camera.keys())
    map_name_to_tuple_key = {image_name: (timestamp, sensor_id)
                             for timestamp, sensor_id, image_name in kapture.flatten(kapture_data.records_camera)}

    number_of_threads = multiprocessing.cpu_count() if max_number_of_threads is None else max_number_of_threads
    with multiprocessing.Pool(number_of_threads) as pool:
        # kapture for localized images + pose
        trajectories = kapture.Trajectories()
        for timestamp in tqdm(timestamps, disable=logging.getLogger().level >= logging.CRITICAL):
            for rig_id in final_camera_list.keys():
                # with S number of sensors
                # N number of correspondences
                # points2D - SxNx2 array with pixel coordinates
                # points3D - SxNx3 array with world coordinates
                # tvec - Sx3 array with rig relative translations
                # qvec - Sx4 array with rig relative quaternions
                # cameras_dict - array of dict of length S
                points2D = []
                points3D = []
                tvec = []
                qvec = []
                cameras_dict = []
                cameras = []  # Sx2 array for reproj error
                stats = []

                for sensor_id, relative_pose in final_camera_list[rig_id].items():
                    if (timestamp, sensor_id) not in kapture_query_data.records_camera:
                        continue
                    img_query = kapture_query_data.records_camera[(timestamp, sensor_id)]
                    if img_query not in pairs:
                        continue

                    st = datetime.datetime.now()
                    keypoints_filepath = keypoints_filepaths[img_query]
                    kapture_keypoints_query = image_keypoints_from_file(
                        filepath=keypoints_filepath,
                        dsize=kapture_data.keypoints[keypoints_type].dsize,
                        dtype=kapture_data.keypoints[keypoints_type].dtype)

                    col_cam_id, width, height, params, _ = get_colmap_camera(
                        kapture_query_data.sensors[sensor_id])

                    st_e = datetime.datetime.now() - st
                    logger.debug(f'preparation of query: {st_e.total_seconds():.3f}')
                    st = datetime.datetime.now()

                    # gather kpid -> matches keypoints for all map images
                    aggregated_matches = aggregate_matches(img_query, pairs[img_query],
                                                           map_name_to_tuple_key,
                                                           keypoints_filepaths,
                                                           kapture_data, keypoints_type,
                                                           kapture_path, tar_handlers)

                    st_e = datetime.datetime.now() - st
                    logger.debug(f'aggregation of matches: {st_e.total_seconds():.3f}')
                    st = datetime.datetime.now()

                    # triangulate all points
                    _, _, _, kpid_to_point3d = \
                        triangulate_all_points(pool,
                                               aggregated_matches,
                                               max_num_iterations,
                                               max_error,
                                               keypoints_type)

                    st_e = datetime.datetime.now() - st
                    logger.debug(f'point triangulation: {st_e.total_seconds():.3f}')
                    st = datetime.datetime.now()

                    if len(kpid_to_point3d) == 0:
                        continue

                    cameras_dict.append({
                        'model': CAMERA_MODEL_NAMES[col_cam_id],
                        'width': int(width),
                        'height': int(height),
                        'params': params
                    })
                    tvec.append(relative_pose.t_raw)
                    qvec.append(relative_pose.r_raw)
                    points2D_it, _, points3D_it, stats_it = convert_correspondences(kpid_to_point3d,
                                                                                    kapture_keypoints_query, None)
                    if write_detailed_report:
                        cameras.append(kapture_query_data.sensors[sensor_id])
                        stats.append(stats_it)
                    points2D.append(points2D_it)
                    points3D.append(points3D_it)

                    st_e = datetime.datetime.now() - st
                    logger.debug(f'convert_correspondences: {st_e.total_seconds():.3f}')
                st = datetime.datetime.now()
                # compute absolute pose
                # inlier_threshold - RANSAC inlier threshold in pixels
                # answer - dictionary containing the RANSAC output
                ret = pycolmap.rig_absolute_pose_estimation(points2D, points3D, cameras_dict, qvec, tvec, max_error,
                                                            min_inlier_ratio, min_num_iterations, max_num_iterations,
                                                            confidence)

                # add pose to output kapture
                if ret['success'] and ret['num_inliers'] > 0:
                    pose = kapture.PoseTransform(ret['qvec'], ret['tvec'])
                    trajectories[timestamp, rig_id] = pose

                    if write_detailed_report:
                        points2D_final = []
                        camera_params = []
                        for points2D_it, query_cam in zip(points2D, cameras):
                            num_2dpoints = len(points2D_it)
                            points2D_final_it, K, distortion = get_camera_matrix_from_kapture(
                                np.array(points2D_it, dtype=np.float), query_cam)
                            points2D_final_it = list(points2D_final_it.reshape((num_2dpoints, 2)))
                            points2D_final.append(points2D_final_it)
                            camera_params.append((K, distortion))
                        num_correspondences = [len(points2D_it) for points2D_it in points2D]
                        # convert ret['inliers']
                        indexes_flat = [i for i, points2D_it in enumerate(points2D) for _ in points2D_it]

                        inliers = [[] for _ in range(len(points2D))]
                        for i, (is_inlier, cam_index) in enumerate(zip(ret['inliers'], indexes_flat)):
                            if is_inlier:
                                inliers[cam_index].append(i)
                        cumulative_len_correspondences = []
                        s = 0
                        for num_correspondences_it in num_correspondences:
                            cumulative_len_correspondences.append(s)
                            s += num_correspondences_it
                        inliers = [[v - cumulative_len_correspondences[i]
                                    for v in inliers[i]] for i in range(len(inliers))]
                        num_inliers = [len(inliers_it) for inliers_it in inliers]

                        per_image_reprojection_error = []
                        for tvec_it, qvec_it, points2D_it, points3D_it, inliers_it, camera_params_it in \
                            zip(tvec, qvec,
                                points2D_final, points3D,
                                inliers, camera_params):
                            if len(inliers_it) == 0:
                                per_image_reprojection_error.append(np.nan)
                            else:
                                pose_relative_it = kapture.PoseTransform(r=qvec_it, t=tvec_it)  # rig to sensor
                                # pose = world to rig
                                pose_it = kapture.PoseTransform.compose([pose_relative_it, pose])  # world to sensor
                                reprojection_error = compute_reprojection_error(
                                    pose_it, len(inliers_it), inliers_it,
                                    points2D_it, points3D_it,
                                    camera_params_it[0], camera_params_it[1])
                                per_image_reprojection_error.append(reprojection_error)

                        cache = {
                            "num_correspondences": num_correspondences,
                            "num_inliers": num_inliers,
                            "inliers": inliers,
                            "reprojection_error": per_image_reprojection_error,
                            "stats": stats
                        }
                        cache_path = os.path.join(output_path, f'pycolmap_rig_cache/{timestamp}.json')
                        save_to_json(cache, cache_path)

                st_e = datetime.datetime.now() - st
                logger.debug(f'pose estimation: {st_e.total_seconds():.3f}')
    # save output kapture
    if apply_rigs_remove:
        rigs_remove_inplace(trajectories, kapture_query_data.rigs)
    kapture_query_data.trajectories = trajectories
    kapture.io.csv.kapture_to_dir(output_path, kapture_query_data)


def get_kapture_pycolmap_rig_localsfm_argparser():
    parser = argparse.ArgumentParser(description=('Localize images from a multi camera rig using pycolmap'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='silently delete output if already exists.')
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory (map + query)')
    parser.add_argument('--query', required=True,
                        help='reference query kapture data root directory')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory.')
    parser.add_argument('--pairsfile-path', required=True,
                        type=str,
                        help=('text file in the csv format; where each line is image_name1, image_name2, score '
                              'which contains the image pairs to match, can be used to filter loaded matches'))

    parser.add_argument('--rig-ids', nargs='+', default=[], help='list of rigs to localize')
    parser.add_argument('--apply-rigs-remove', action='store_true', default=False,
                        help='apply rigs remove before saving poses to disk.')

    parser.add_argument('--max-error', type=float, default=8.0,
                        help='RANSACOptions max_error, in pixels. Use about 1 percent of images diagonal')
    parser.add_argument('--min-inlier-ratio', type=int, default=0.01,
                        help='abs_pose_options.ransac_options.min_inlier_ratio')
    parser.add_argument('--min-num-iterations', type=int, default=1000,
                        help='abs_pose_options.ransac_options.min_num_trials')
    parser.add_argument('--max-num-iterations', type=int, default=100000,
                        help='abs_pose_options.ransac_options.max_num_trials')
    parser.add_argument('--confidence', type=int, default=0.9999,
                        help='abs_pose_options.ransac_options.confidence')

    parser.add_argument('--keypoints-type', default=None,  help='keypoint type_name')
    parser.add_argument('--write-detailed-report', action='store_true', default=False,
                        help='write inliers and reprojection error in a json for each query.')

    parser.add_argument('--max-number-of-threads', default=None, type=int,
                        help='By default, use as many as cpus. But you can set a limit.')
    return parser


def kapture_pycolmap_rig_localsfm_command_line():
    """
    Parse the command line arguments to localize images
    from a multi camera rig with pycolmap using the given kapture data.
    """
    parser = get_kapture_pycolmap_rig_localsfm_argparser()
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    getLogger().setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('kapture_pycolmap_rig_localsfm.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    kapture_pycolmap_rig_localsfm(args.input,
                                  args.query,
                                  args.output,
                                  args.pairsfile_path,
                                  args.rig_ids,
                                  args.apply_rigs_remove,
                                  args.max_error,
                                  args.min_inlier_ratio,
                                  args.min_num_iterations,
                                  args.max_num_iterations,
                                  args.confidence,
                                  args.keypoints_type,
                                  args.write_detailed_report,
                                  args.max_number_of_threads,
                                  args.force)


if __name__ == '__main__':
    kapture_pycolmap_rig_localsfm_command_line()
