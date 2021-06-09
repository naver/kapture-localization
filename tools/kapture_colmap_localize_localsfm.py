#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import argparse
import logging
import itertools
from collections import OrderedDict
from typing import Optional

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.path_to_kapture  # noqa: F401

from kapture_colmap_build_map import colmap_build_map_from_loaded_data
from kapture_colmap_localize import colmap_localize_from_loaded_data
from kapture_run_colmap_gv import run_colmap_gv_from_loaded_data
from kapture_compute_matches import compute_matches_from_loaded_data

import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, kapture_to_dir, table_from_file, table_to_file, get_all_tar_handlers
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.io.records import TransferAction
from kapture.converter.colmap.import_colmap import import_colmap
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
from kapture.io.tar import TarCollection, TarHandler, retrieve_tar_handler_from_collection, get_feature_tar_fullpath
from kapture.utils.Collections import try_get_only_key_from_collection

logger = logging.getLogger()


def sub_kapture_from_img_list(kdata, img_list, pairs, keypoints_type, descriptors_type):
    trajectories = kapture.Trajectories()
    sensors = kapture.Sensors()
    records = kapture.RecordsCamera()
    keypoints = kapture.Keypoints(kdata.keypoints[keypoints_type].type_name,
                                  kdata.keypoints[keypoints_type].dtype,
                                  kdata.keypoints[keypoints_type].dsize)
    if kdata.descriptors is not None and descriptors_type in kdata.descriptors:
        descriptors = kapture.Descriptors(kdata.descriptors[descriptors_type].type_name,
                                          kdata.descriptors[descriptors_type].dtype,
                                          kdata.descriptors[descriptors_type].dsize,
                                          kdata.descriptors[descriptors_type].keypoints_type,
                                          kdata.descriptors[descriptors_type].metric_type)
    else:
        descriptors = None
    matches = kapture.Matches()

    timestamp_sensor_id_from_image_name = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                           kapture.flatten(kdata.records_camera)}
    for img in img_list:
        timestamp, sensor_id = timestamp_sensor_id_from_image_name[img]
        sensors[sensor_id] = kdata.sensors[sensor_id]
        records[timestamp, sensor_id] = img
        if (timestamp, sensor_id) in kdata.trajectories:
            pose = kdata.trajectories[timestamp][sensor_id]
            trajectories[timestamp, sensor_id] = pose
        keypoints.add(img)
        if kdata.descriptors is not None:
            descriptors.add(img)

    for i in pairs:
        if i in kdata.matches[keypoints_type]:
            matches.add(i[0], i[1])
    matches.normalize()

    return kapture.Kapture(sensors=sensors, trajectories=trajectories, records_camera=records,
                           descriptors={descriptors_type: descriptors},
                           keypoints={keypoints_type: keypoints},
                           matches={keypoints_type: matches})


def add_image_to_kapture(kdata_src,
                         kdata_trg, img_name, pairs,
                         keypoints_type, descriptors_type,
                         add_pose=False):
    timestamp_sensor_id_from_image_name = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                           kapture.flatten(kdata_src.records_camera)}
    timestamp, sensor_id = timestamp_sensor_id_from_image_name[img_name]
    kdata_trg.sensors[sensor_id] = kdata_src.sensors[sensor_id]
    kdata_trg.records_camera[timestamp, sensor_id] = img_name
    kdata_trg.keypoints[keypoints_type].add(img_name)
    if kdata_trg.descriptors is not None and descriptors_type in kdata_trg.descriptors:
        kdata_trg.descriptors[descriptors_type].add(img_name)

    if add_pose:
        kdata_trg.trajectories[timestamp, sensor_id] = kdata_src.trajectories[timestamp, sensor_id]

    if len(pairs) != 0:
        if kdata_trg.matches is None:
            kdata_trg.matches = {}
        kdata_trg.matches[keypoints_type] = kapture.Matches()
        for i in pairs:
            if i in kdata_src.matches[keypoints_type]:
                kdata_trg.matches[keypoints_type].add(i[0], i[1])
        kdata_trg.matches[keypoints_type].normalize()
    return kdata_trg


def pose_found(kdata, img_name):
    timestamp_sensor_id_from_image_name = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                           kapture.flatten(kdata.records_camera)}
    if img_name in timestamp_sensor_id_from_image_name:
        timestamp, sensor_id = timestamp_sensor_id_from_image_name[img_name]
        if (timestamp, sensor_id) in kdata.trajectories:
            return True
        else:
            return False
    else:
        logger.info(f'something is wrong with {img_name} (it will be skipped), check input kapture data')
        return True


def add_pose_to_query_kapture(kdata_src, kdata_trg, img_name):
    timestamp_sensor_id_from_image_name_src = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                               kapture.flatten(kdata_src.records_camera)}
    if img_name not in timestamp_sensor_id_from_image_name_src:
        logger.info(f'{img_name} was not found in localized kapture, that should not be possible, something went wrong')
        return False
    timestamp_src, sensor_id_src = timestamp_sensor_id_from_image_name_src[img_name]
    timestamp_sensor_id_from_image_name_trg = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                               kapture.flatten(kdata_trg.records_camera)}
    if img_name not in timestamp_sensor_id_from_image_name_trg:
        logger.info(f'{img_name} not found in query kapture')
        return False
    timestamp_trg, sensor_id_trg = timestamp_sensor_id_from_image_name_trg[img_name]

    if not (timestamp_src, sensor_id_src) in kdata_src.trajectories:
        logger.info(f'{img_name} was not localized')
        return False
    kdata_trg.trajectories[timestamp_trg, sensor_id_trg] = kdata_src.trajectories[timestamp_src, sensor_id_src]

    return True


def get_pairfile_from_img_list(img_list):
    image_pairs = [(i, j, 0) if i < j else (j, i, 0) for i, j in itertools.product(img_list, img_list)]
    image_pairs = list(OrderedDict.fromkeys(image_pairs))
    image_pairs_filtered = []
    for i in image_pairs:
        if i[0] != i[1]:
            image_pairs_filtered.append(i)
    return image_pairs_filtered


def get_pairfile_img_vs_img_list(img, img_list):
    image_pairs = []
    for i in img_list:
        if img < i:
            image_pairs.append((img, i, 0))
        else:
            image_pairs.append((i, img, 0))
    return image_pairs


def local_sfm(map_plus_query_path: str,
              map_plus_query_gv_path: str,
              query_path: str,
              descriptors_type: Optional[str],
              pairsfile_path: str,
              output_path_root: str,
              colmap_binary: str,
              force: bool):
    """
    Localize query images in a COLMAP model built from topk retrieved images.

    :param map_plus_query_path: path to the kapture data consisting of mapping and query data (sensors and reconstruction)
    :param map_plus_query_gv_path: path to the kapture data consisting of mapping and query data after geometric verification (sensors and reconstruction)
    :param query_path: path to the query kapture data (sensors)
    :param descriptors_type: type of descriptors, name of the descriptors subfolder
    :param pairsfile_path: path to the pairsfile that contains the topk retrieved mapping images for each query image
    :param output_path_root: root path where outputs should be stored
    :param colmap_binary: path to the COLMAP binary
    :param force: silently overwrite already existing results
    """
    kdata_query = kapture_from_dir(query_path)
    with get_all_tar_handlers(map_plus_query_path,
                              mode={kapture.Keypoints: 'r',
                                    kapture.Descriptors: 'r',
                                    kapture.GlobalFeatures: 'r',
                                    kapture.Matches: 'a'}) as tar_handlers_map:
        kdata_map = kapture_from_dir(map_plus_query_path, tar_handlers=tar_handlers_map)
        with get_all_tar_handlers(map_plus_query_gv_path,
                                  mode={kapture.Keypoints: 'r',
                                        kapture.Descriptors: 'r',
                                        kapture.GlobalFeatures: 'r',
                                        kapture.Matches: 'a'}) as tar_handlers_map_gv:
            kdata_map_gv = kapture_from_dir(map_plus_query_gv_path, tar_handlers=tar_handlers_map_gv)
            local_sfm_from_loaded_data(kdata_map, kdata_map_gv, kdata_query,
                                       map_plus_query_path, map_plus_query_gv_path,
                                       tar_handlers_map,
                                       tar_handlers_map_gv,
                                       descriptors_type,
                                       pairsfile_path,
                                       output_path_root,
                                       colmap_binary,
                                       force)


def local_sfm_from_loaded_data(kdata_map: kapture.Kapture,
                               kdata_map_gv: kapture.Kapture,
                               kdata_query: kapture.Kapture,
                               map_plus_query_path: str,
                               map_plus_query_gv_path: str,
                               tar_handlers_map: Optional[TarCollection],
                               tar_handlers_map_gv: Optional[TarCollection],
                               descriptors_type: Optional[str],
                               pairsfile_path: str,
                               output_path_root: str,
                               colmap_binary: str,
                               force: bool):
    """
    Localize query images in a COLMAP model built from topk retrieved images.

    :param map_plus_query_path: path to the kapture data consisting of mapping and query data (sensors and reconstruction)
    :param map_plus_query_gv_path: path to the kapture data consisting of mapping and query data after geometric verification (sensors and reconstruction)
    :param query_path: path to the query kapture data (sensors)
    :param descriptors_type: type of descriptors, name of the descriptors subfolder
    :param pairsfile_path: path to the pairsfile that contains the topk retrieved mapping images for each query image
    :param output_path_root: root path where outputs should be stored
    :param colmap_binary: path to the COLMAP binary
    :param force: silently overwrite already existing results
    """

    # load query kapture (we use query kapture to reuse sensor_ids etc.)
    if kdata_query.trajectories:
        logger.warning("Query data contains trajectories: they will be ignored")
        kdata_query.trajectories.clear()
    else:
        kdata_query.trajectories = kapture.Trajectories()

    # clear query trajectories in map_plus_query
    kdata_map_cleared_trajectories = kapture.Trajectories()
    query_image_list = set(kdata_query.records_camera.data_list())
    for timestamp, subdict in kdata_map.records_camera.items():
        for sensor_id, image_name in subdict.items():
            if image_name in query_image_list:
                continue
            if (timestamp, sensor_id) in kdata_map.trajectories:
                pose = kdata_map.trajectories.get(timestamp)[sensor_id]
                kdata_map_cleared_trajectories.setdefault(timestamp, {})[sensor_id] = pose
    kdata_map.trajectories = kdata_map_cleared_trajectories

    # load output kapture
    output_path = os.path.join(output_path_root, 'localized')
    if os.path.exists(os.path.join(output_path, 'sensors/trajectories.txt')):
        kdata_output = kapture_from_dir(output_path)
        if kdata_query.records_camera == kdata_output.records_camera and len(
                kdata_output.trajectories) != 0 and not force:
            kdata_query.trajectories = kdata_output.trajectories

    if kdata_map.rigs is not None:
        rigs_remove_inplace(kdata_map.trajectories, kdata_map.rigs)
    if kdata_map_gv.rigs is not None:
        rigs_remove_inplace(kdata_map_gv.trajectories, kdata_map_gv.rigs)

    # load pairsfile
    pairs = {}
    with open(pairsfile_path, 'r') as fid:
        table = table_from_file(fid)
        for img_query, img_map, _ in table:
            if img_query not in pairs:
                pairs[img_query] = []
            pairs[img_query].append(img_map)

    kdata_sub_colmap_path = os.path.join(output_path_root, 'colmap')
    kdata_reg_query_path = os.path.join(output_path_root, 'query_registered')
    sub_kapture_pairsfile_path = os.path.join(output_path_root, 'tmp_pairs.txt')

    if descriptors_type is None:
        descriptors_type = try_get_only_key_from_collection(kdata_map.descriptors)
    assert descriptors_type is not None
    assert descriptors_type in kdata_map.descriptors
    keypoints_type = kdata_map.descriptors[descriptors_type].keypoints_type

    # init matches for kdata_map and kdata_map_gv
    if kdata_map.matches is None:
        kdata_map.matches = {}
    if keypoints_type not in kdata_map.matches:
        kdata_map.matches[keypoints_type] = kapture.Matches()
    if kdata_map_gv.matches is None:
        kdata_map_gv.matches = {}
    if keypoints_type not in kdata_map_gv.matches:
        kdata_map_gv.matches[keypoints_type] = kapture.Matches()

    # run all matching
    # loop over query images
    img_skip_list = set()
    for img_query, img_list_map in pairs.items():
        if pose_found(kdata_query, img_query):
            logger.info(f'{img_query} already processed, skipping...')
            img_skip_list.add(img_query)
            continue
        else:
            map_pairs = get_pairfile_from_img_list(img_list_map)
            query_pairs = get_pairfile_img_vs_img_list(img_query, img_list_map)
            with open(sub_kapture_pairsfile_path, 'w') as fid:
                logger.info(f'matching for {img_query}')
                table_to_file(fid, map_pairs)
                table_to_file(fid, query_pairs)

            pairs_all = map_pairs + query_pairs
            pairs_all = [(i, j) for i, j, _ in pairs_all]
            # match missing pairs
            # kdata_map.matches is being updated by compute_matches_from_loaded_data
            compute_matches_from_loaded_data(map_plus_query_path,
                                             tar_handlers_map,
                                             kdata_map,
                                             descriptors_type,
                                             pairs_all)

    # if kdata_map have matches in tar, they need to be switched to read mode
    matches_handler = retrieve_tar_handler_from_collection(kapture.Matches, keypoints_type, tar_handlers_map)
    if matches_handler is not None:
        matches_handler.close()
        tarfile_path = get_feature_tar_fullpath(kapture.Matches, keypoints_type, map_plus_query_path)
        tar_handlers_map.matches[keypoints_type] = TarHandler(tarfile_path, 'r')

    # run all gv
    # loop over query images
    for img_query, img_list_map in pairs.items():
        if img_query in img_skip_list:
            continue
        else:
            # recompute the pairs
            map_pairs = get_pairfile_from_img_list(img_list_map)
            query_pairs = get_pairfile_img_vs_img_list(img_query, img_list_map)
            with open(sub_kapture_pairsfile_path, 'w') as fid:
                logger.info(f'geometric verification of {img_query}')
                table_to_file(fid, map_pairs)
                table_to_file(fid, query_pairs)

            pairs_all = map_pairs + query_pairs
            pairs_all = [(i, j) for i, j, _ in pairs_all]

            if all(pair in kdata_map_gv.matches[keypoints_type] for pair in pairs_all):
                continue

            # create a sub kapture in order to minimize the amount of data exported to colmap
            # kdata_sub needs to be re-created to add the new matches
            kdata_sub = sub_kapture_from_img_list(kdata_map, img_list_map + [img_query], pairs_all,
                                                  keypoints_type, descriptors_type)

            kdata_sub_gv = sub_kapture_from_img_list(kdata_map_gv, img_list_map + [img_query], pairs_all,
                                                     keypoints_type, descriptors_type)
            # run colmap gv on missing pairs
            run_colmap_gv_from_loaded_data(kdata_sub,
                                           kdata_sub_gv,
                                           map_plus_query_path,
                                           map_plus_query_gv_path,
                                           tar_handlers_map,
                                           tar_handlers_map_gv,
                                           colmap_binary,
                                           keypoints_type,
                                           [],
                                           True)
            # update kdata_map_gv.matches
            kdata_map_gv.matches[keypoints_type].update(kdata_sub_gv.matches[keypoints_type])

    # if kdata_map_gv have matches in tar, they need to be switched to read mode
    matches_gv_handler = retrieve_tar_handler_from_collection(kapture.Matches, keypoints_type, tar_handlers_map_gv)
    if matches_gv_handler is not None:
        print(matches_gv_handler)
        matches_gv_handler.close()
        tarfile_path = get_feature_tar_fullpath(kapture.Matches, keypoints_type, map_plus_query_gv_path)
        tar_handlers_map_gv.matches[keypoints_type] = TarHandler(tarfile_path, 'r')

    # loop over query images
    for img_query, img_list_map in pairs.items():
        if img_query in img_skip_list:
            continue
        else:
            map_pairs = get_pairfile_from_img_list(img_list_map)
            with open(sub_kapture_pairsfile_path, 'w') as fid:
                logger.info(f'mapping and localization for {img_query}')
                table_to_file(fid, map_pairs)
            map_pairs = [(i, j) for i, j, _ in map_pairs]
            kdata_sub_gv = sub_kapture_from_img_list(kdata_map_gv, img_list_map, map_pairs,
                                                     keypoints_type, descriptors_type)
            # sanity check
            if len(map_pairs) != len(kdata_sub_gv.matches[keypoints_type]):
                logger.info(f'not all mapping matches available')

            # build COLMAP map
            try:
                colmap_build_map_from_loaded_data(
                    kdata_sub_gv,
                    map_plus_query_gv_path,
                    tar_handlers_map_gv,
                    kdata_sub_colmap_path,
                    colmap_binary,
                    keypoints_type,
                    False,
                    [],
                    ['model_converter'],
                    True)
            except ValueError:
                logger.info(f'{img_query} was not localized')
                continue

        if not os.path.exists(os.path.join(kdata_sub_colmap_path, 'reconstruction/images.bin')):
            logger.info(f'colmap mapping for {img_query} did not work, image was not localized')
            continue

        query_pairs = get_pairfile_img_vs_img_list(img_query, img_list_map)
        with open(sub_kapture_pairsfile_path, 'w') as fid:
            table_to_file(fid, query_pairs)
        query_pairs = [(i, j) for i, j, _ in query_pairs]
        query_img_kapture_gv = add_image_to_kapture(kdata_map_gv,
                                                    kdata_sub_gv, img_query, query_pairs,
                                                    keypoints_type, descriptors_type)
        # sanity check
        if len(query_pairs) != len(query_img_kapture_gv.matches[keypoints_type]):
            logger.info(f'not all query matches available')

        # localize in COLMAP map
        try:
            colmap_localize_from_loaded_data(
                query_img_kapture_gv,
                map_plus_query_gv_path,
                tar_handlers_map_gv,
                os.path.join(kdata_sub_colmap_path, 'registered'),
                os.path.join(kdata_sub_colmap_path, 'colmap.db'),
                os.path.join(kdata_sub_colmap_path, 'reconstruction'),
                colmap_binary,
                keypoints_type,
                False,
                ['--Mapper.ba_refine_focal_length', '0',
                 '--Mapper.ba_refine_principal_point', '0',
                 '--Mapper.ba_refine_extra_params', '0',
                 '--Mapper.min_num_matches', '4',
                 '--Mapper.init_min_num_inliers', '4',
                 '--Mapper.abs_pose_min_num_inliers', '4',
                 '--Mapper.abs_pose_min_inlier_ratio', '0.05',
                 '--Mapper.ba_local_max_num_iterations', '50',
                 '--Mapper.abs_pose_max_error', '20',
                 '--Mapper.filter_max_reproj_error', '12'],
                [],
                True)
        except ValueError:
            logger.info(f'{img_query} was not localized')
            continue

        if not os.path.exists(os.path.join(os.path.join(kdata_sub_colmap_path, 'registered'),
                                           'reconstruction/images.txt')):
            logger.info(f'colmap localization of {img_query} did not work, image was not localized')
            continue

        # add to results kapture
        kdata_reg_query = import_colmap(
            kdata_reg_query_path,
            os.path.join(os.path.join(kdata_sub_colmap_path, 'registered'), 'colmap.db'),
            os.path.join(os.path.join(kdata_sub_colmap_path, 'registered'),
                         'reconstruction'),
            None,
            None,
            True,
            True,
            True,
            TransferAction.skip)

        if add_pose_to_query_kapture(kdata_reg_query, kdata_query, img_query):
            logger.info('successfully localized')

        # write results (after each image to see the progress)
        kapture_to_dir(output_path, kdata_query)

    # clean up (e.g. remove temporal files and folders)
    safe_remove_any_path(kdata_sub_colmap_path, True)
    safe_remove_any_path(kdata_reg_query_path, True)
    safe_remove_file(sub_kapture_pairsfile_path, True)

    logger.info('all done')


def local_sfm_command_line():
    parser = argparse.ArgumentParser(
        description='local sfm localization with COLMAP: each query image has topk db images (defined in one pairsfile)'
                    'these db images are used to build a COLMAP model'
                    'then the query image is localized in this model')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force recomputation of existing data')
    parser.add_argument('--map_plus_query', required=True,
                        help='input path to kapture containing mapping and query data (sensors and reconstruction)')
    parser.add_argument('--map_plus_query_gv', required=True,
                        help='input path to kapture containing mapping and query data after geometric verification (sensors and reconstruction)')
    parser.add_argument('--query', required=True,
                        help='path to kapture containing query data (sensors)')
    parser.add_argument('-o', '--output', required=True,
                        help='output kapture directory')
    parser.add_argument('-colmap', '--colmap_binary', required=False,
                        default="colmap",
                        help='full path to colmap binary '
                             '(default is "colmap", i.e. assume the binary'
                             ' is in the user PATH).')
    parser.add_argument('--pairsfile-path',
                        default=None,
                        type=str,
                        help='text file containing the image pairs between query and mapping images')
    parser.add_argument('-desc', '--descriptors-type', default=None, help='kapture descriptors type.')

    args = parser.parse_args()

    logger.setLevel(args.verbose)
    logging.getLogger('colmap').setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    local_sfm(args.map_plus_query, args.map_plus_query_gv, args.query,
              args.descriptors_type,
              args.pairsfile_path, args.output,
              args.colmap_binary, args.force)


if __name__ == '__main__':
    local_sfm_command_line()
