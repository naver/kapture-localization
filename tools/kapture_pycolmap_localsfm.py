#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script localize images using pycolmap from kapture format
"""

import argparse
import logging
import os
from tqdm import tqdm
from typing import Optional
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
from kapture_localization.triangulation.integration import aggregate_matches, triangulate_all_points
from kapture_localization.triangulation.integration import convert_correspondences

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.io.csv
import kapture.utils.logging
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.io.features import keypoints_to_filepaths, image_keypoints_from_file
from kapture.io.structure import delete_existing_kapture_files
from kapture.converter.colmap.cameras import get_colmap_camera, CAMERA_MODEL_NAME_ID
from kapture.utils.Collections import try_get_only_key_from_collection
from kapture.io.tar import TarCollection


logger = logging.getLogger('kapture_pycolmap_localsfm')


def kapture_pycolmap_localsfm(kapture_path: str,
                              kapture_query_path: str,
                              output_path: str,
                              pairsfile_path: str,
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
    Localize images using pycolmap.

    :param kapture_path: path to the kapture to use (mapping and query images)
    :param kapture_query_path: path to the kapture to use (query images)
    :param output_path: path to the write the localization results
    :param pairsfile_path: pairs to use
    :param max_error: RANSAC inlier threshold in pixel
    :param min_inlier_ratio: abs_pose_options.ransac_options.min_inlier_ratio
    :param min_num_iterations: abs_pose_options.ransac_options.min_num_trials
    :param max_num_iterations: abs_pose_options.ransac_options.max_num_trials
    :param confidence: abs_pose_options.ransac_options.confidence
    :param keypoints_type: types of keypoints (and observations) to use
    :param write_detailed_report: if True, write a json file with inliers, reprojection error for each query
    :param max_number_of_threads: maximum number of parallel triangulations running
    :param force: Silently overwrite kapture files if already exists
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

        kapture_pycolmap_localsfm_from_loaded_data(kapture_data,
                                                   kapture_path,
                                                   tar_handlers,
                                                   kapture_query_data,
                                                   output_path,
                                                   pairsfile_path,
                                                   max_error,
                                                   min_inlier_ratio,
                                                   min_num_iterations,
                                                   max_num_iterations,
                                                   confidence,
                                                   keypoints_type,
                                                   write_detailed_report,
                                                   max_number_of_threads,
                                                   force)


def kapture_pycolmap_localsfm_from_loaded_data(kapture_data: kapture.Kapture,
                                               kapture_path: str,
                                               tar_handlers: TarCollection,
                                               kapture_query_data: kapture.Kapture,
                                               output_path: str,
                                               pairsfile_path: str,
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
    Localize images using pycolmap.

    :param kapture_data: loaded kapture data (mapping and query images, incl. matches)
    :param kapture_path: path to the kapture to use
    :param tar_handlers: collection of pre-opened tar archives
    :param kapture_query_data: loaded kapture data (query images)
    :param output_path: path to the write the localization results
    :param pairsfile_path: pairs to use
    :param max_error: RANSAC inlier threshold in pixel
    :param min_inlier_ratio: abs_pose_options.ransac_options.min_inlier_ratio
    :param min_num_iterations: abs_pose_options.ransac_options.min_num_trials
    :param max_num_iterations: abs_pose_options.ransac_options.max_num_trials
    :param confidence: abs_pose_options.ransac_options.confidence
    :param keypoints_type: types of keypoints (and observations) to use
    :param write_detailed_report: if True, write a json file with inliers, reprojection error for each query
    :param max_number_of_threads: maximum number of parallel triangulations running
    :param force: Silently overwrite kapture files if already exists
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

    if kapture_data.rigs is not None and kapture_data.trajectories is not None:
        # make sure, rigs are not used in trajectories.
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

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

    query_images = [(timestamp, sensor_id, image_name)
                    for timestamp, sensor_id, image_name in kapture.flatten(kapture_query_data.records_camera)]
    map_name_to_tuple_key = {image_name: (timestamp, sensor_id)
                             for timestamp, sensor_id, image_name in kapture.flatten(kapture_data.records_camera)}

    number_of_threads = multiprocessing.cpu_count() if max_number_of_threads is None else max_number_of_threads
    with multiprocessing.Pool(number_of_threads) as pool:
        # kapture for localized images + pose
        trajectories = kapture.Trajectories()
        for timestamp, sensor_id, image_name in tqdm(query_images,
                                                     disable=logging.getLogger().level >= logging.CRITICAL):
            if image_name not in pairs:
                continue

            st = datetime.datetime.now()
            keypoints_filepath = keypoints_filepaths[image_name]
            kapture_keypoints_query = image_keypoints_from_file(filepath=keypoints_filepath,
                                                                dsize=kapture_data.keypoints[keypoints_type].dsize,
                                                                dtype=kapture_data.keypoints[keypoints_type].dtype)
            query_cam = kapture_query_data.sensors[sensor_id]
            assert isinstance(query_cam, kapture.Camera)
            col_cam_id, width, height, params, _ = get_colmap_camera(query_cam)
            cfg = {
                'model': CAMERA_MODEL_NAME_ID[col_cam_id][0],
                'width': int(width),
                'height': int(height),
                'params': params
            }
            st_e = datetime.datetime.now() - st
            logger.debug(f'preparation of query: {st_e.total_seconds():.3f}')
            st = datetime.datetime.now()
            # gather kpid -> matches keypoints for all map images
            aggregated_matches = aggregate_matches(image_name, pairs[image_name],
                                                   map_name_to_tuple_key,
                                                   keypoints_filepaths,
                                                   kapture_data, keypoints_type,
                                                   kapture_path, tar_handlers)

            st_e = datetime.datetime.now() - st
            logger.debug(f'aggregation of matches: {st_e.total_seconds():.3f}')
            st = datetime.datetime.now()

            # triangulate all points
            _, _, _, kpid_to_point3d = triangulate_all_points(
                pool,
                aggregated_matches,
                max_num_iterations,
                max_error,
                keypoints_type
            )

            st_e = datetime.datetime.now() - st
            logger.debug(f'point triangulation: {st_e.total_seconds():.3f}')
            st = datetime.datetime.now()

            if len(kpid_to_point3d) == 0:
                continue

            points2D, _, points3D, stats = convert_correspondences(kpid_to_point3d,
                                                                   kapture_keypoints_query, None)

            st_e = datetime.datetime.now() - st
            logger.debug(f'convert_correspondences: {st_e.total_seconds():.3f}')
            st = datetime.datetime.now()

            # compute absolute pose
            # inlier_threshold - RANSAC inlier threshold in pixels
            # answer - dictionary containing the RANSAC output
            ret = pycolmap.absolute_pose_estimation(points2D, points3D, cfg, max_error,
                                                    min_inlier_ratio, min_num_iterations,
                                                    max_num_iterations, confidence)

            # add pose to output kapture
            if ret['success'] and ret['num_inliers'] > 0:
                pose = kapture.PoseTransform(ret['qvec'], ret['tvec'])
                if write_detailed_report:
                    num_2dpoints = len(points2D)
                    points2D_final, K, distortion = get_camera_matrix_from_kapture(
                        np.array(points2D, dtype=np.float), query_cam)
                    points2D_final = list(points2D_final.reshape((num_2dpoints, 2)))
                    inliers = np.where(ret['inliers'])[0].tolist()
                    reprojection_error = compute_reprojection_error(pose, ret['num_inliers'], inliers,
                                                                    points2D_final, points3D, K, distortion)
                    cache = {
                        "num_correspondences": len(points3D),
                        "num_inliers": inliers,
                        "inliers": ret['inliers'],
                        "reprojection_error": reprojection_error,
                        "stats": stats
                    }
                    cache_path = os.path.join(output_path, f'pycolmap_cache/{image_name}.json')
                    save_to_json(cache, cache_path)
                trajectories[timestamp, sensor_id] = pose

            st_e = datetime.datetime.now() - st
            logger.debug(f'pose estimation: {st_e.total_seconds():.3f}')

    kapture_data_localized = kapture.Kapture(sensors=kapture_query_data.sensors,
                                             trajectories=trajectories,
                                             records_camera=kapture_query_data.records_camera,
                                             rigs=kapture_query_data.rigs)
    kapture.io.csv.kapture_to_dir(output_path, kapture_data_localized)


def get_kapture_pycolmap_localsfm_argparser():
    parser = argparse.ArgumentParser(description=('localize images with pycolmap '
                                                  'from data specified in kapture format.'))
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


def kapture_pycolmap_localsfm_command_line():
    """
    Parse the command line arguments to localize images with pycolmap using the given kapture data.
    """
    parser = get_kapture_pycolmap_localsfm_argparser()
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    getLogger().setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('kapture_pycolmap_localsfm.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    kapture_pycolmap_localsfm(args.input,
                              args.query,
                              args.output,
                              args.pairsfile_path,
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
    kapture_pycolmap_localsfm_command_line()
