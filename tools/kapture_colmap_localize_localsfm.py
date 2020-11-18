#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import argparse
import logging
import itertools
from collections import OrderedDict

import path_to_kapture_localization
import kapture_localization.utils.path_to_kapture

from kapture_colmap_build_map import colmap_build_map_from_loaded_data
from kapture_colmap_localize import colmap_localize_from_loaded_data
from kapture_run_colmap_gv import run_colmap_gv_from_loaded_data
from kapture_compute_matches import compute_matches_from_loaded_data

import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, kapture_to_dir, table_from_file
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.io.features import get_matches_fullpath
from kapture.io.records import TransferAction
from kapture.converter.colmap.import_colmap import import_colmap
from kapture.utils.paths import safe_remove_file, safe_remove_any_path

logger = logging.getLogger()


def sub_kapture_from_img_list(kdata, kdata_path, img_list, pairs):
    trajectories = kapture.Trajectories()
    sensors = kapture.Sensors()
    records = kapture.RecordsCamera()
    keypoints = kapture.Keypoints(kdata.keypoints._tname, kdata.keypoints._dtype, kdata.keypoints._dsize)
    if kdata.descriptors != None:
        descriptors = kapture.Descriptors(kdata.descriptors._tname, kdata.descriptors._dtype, kdata.descriptors._dsize)
    else:
        descriptors = None
    matches = kapture.Matches()

    timestamp_sensor_id_from_image_name = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                           kapture.flatten(kdata.records_camera)}
    for img in img_list:
        timestamp, sensor_id = timestamp_sensor_id_from_image_name[img]
        pose = kdata.trajectories[timestamp][sensor_id]
        sensors[sensor_id] = kdata.sensors[sensor_id]
        records[timestamp, sensor_id] = img
        trajectories[timestamp, sensor_id] = pose
        keypoints.add(img)
        if kdata.descriptors != None:
            descriptors.add(img)

    for i in pairs:
        image_matches_filepath = get_matches_fullpath((i[0], i[1]), kdata_path)
        if os.path.exists(image_matches_filepath):
            matches.add(i[0], i[1])
    matches.normalize()

    return kapture.Kapture(sensors=sensors, trajectories=trajectories, records_camera=records, descriptors=descriptors,
                           keypoints=keypoints, matches=matches)


def add_image_to_kapture(kdata_src, kdata_src_path, kdata_trg, img_name, pairs, add_pose=False):
    timestamp_sensor_id_from_image_name = {img_name: (timestamp, sensor_id) for timestamp, sensor_id, img_name in
                                           kapture.flatten(kdata_src.records_camera)}
    timestamp, sensor_id = timestamp_sensor_id_from_image_name[img_name]
    kdata_trg.sensors[sensor_id] = kdata_src.sensors[sensor_id]
    kdata_trg.records_camera[timestamp, sensor_id] = img_name
    kdata_trg.keypoints.add(img_name)
    if kdata_trg.descriptors != None:
        kdata_trg.descriptors.add(img_name)

    if add_pose:
        kdata_trg.trajectories[timestamp, sensor_id] = kdata_src.trajectories[timestamp, sensor_id]

    if os.path.exists(kdata_src_path) and len(pairs) != 0:
        kdata_trg.matches = kapture.Matches()
        for i in pairs:
            image_matches_filepath = get_matches_fullpath((i[0], i[1]), kdata_src_path)
            if os.path.exists(image_matches_filepath):
                kdata_trg.matches.add(i[0], i[1])
        kdata_trg.matches.normalize()

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


def write_pairfile_from_img_list(img_list, pairsfile_path):
    image_pairs = [(i, j) if i < j else (j, i) for i, j in itertools.product(img_list, img_list)]
    image_pairs = list(OrderedDict.fromkeys(image_pairs))
    image_pairs_filtered = []
    for i in image_pairs:
        if i[0] != i[1]:
            image_pairs_filtered.append(i)
    image_pairs = []
    with open(pairsfile_path, 'w') as file:
        for i in image_pairs_filtered:
            if i[0] < i[1]:
                file.writelines(f'{i[0]}, {i[1]}, 0\n')
                image_pairs.append((i[0], i[1]))
            else:
                file.writelines(f'{i[1]}, {i[0]}, 0\n')
                image_pairs.append((i[1], i[0]))
    return image_pairs


def write_pairfile_img_vs_img_list(img, img_list, pairsfile_path):
    image_pairs = []
    with open(pairsfile_path, 'w') as file:
        for i in img_list:
            if img < i:
                file.writelines(f'{img}, {i}, 0\n')
                image_pairs.append((img, i))
            else:
                file.writelines(f'{i}, {img}, 0\n')
                image_pairs.append((i, img))
    return image_pairs


def local_sfm(map_plus_query_path: str,
              map_plus_query_gv_path: str,
              query_path: str,
              pairsfile_path: str,
              output_path_root: str,
              colmap_binary: str,
              force: bool):
    """
    Localize query images in a COLMAP model built from topk retrieved images.

    :param map_plus_query_path: path to the kapture data consisting of mapping and query data (sensors and reconstruction)
    :param map_plus_query_gv_path: path to the kapture data consisting of mapping and query data after geometric verification (sensors and reconstruction)
    :param query_path: path to the query kapture data (sensors)
    :param pairsfile_path: path to the pairsfile that contains the topk retrieved mapping images for each query image
    :param output_path_root: root path where outputs should be stored
    :param colmap_binary: path to the COLMAP binary
    :param force: silently overwrite already existing results
    """

    # load query kapture (we use query kapture to reuse sensor_ids etc.)
    kdata_query = kapture_from_dir(query_path)
    if kdata_query.trajectories:
        logger.warning("Query data contains trajectories: they will be ignored")
        kdata_query.trajectories.clear()
    else:
        kdata_query.trajectories = kapture.Trajectories()

    # load output kapture
    output_path = os.path.join(output_path_root, 'localized')
    if os.path.exists(os.path.join(output_path, 'sensors/trajectories.txt')):
        kdata_output = kapture_from_dir(output_path)
        if kdata_query.records_camera == kdata_output.records_camera and len(
            kdata_output.trajectories) != 0 and not force:
            kdata_query.trajectories = kdata_output.trajectories

    # load kapture maps
    kdata_map = kapture_from_dir(map_plus_query_path)
    if kdata_map.rigs != None:
        rigs_remove_inplace(kdata_map.trajectories, kdata_map.rigs)
    kdata_map_gv = kapture_from_dir(map_plus_query_gv_path)
    if kdata_map_gv.rigs != None:
        rigs_remove_inplace(kdata_map_gv.trajectories, kdata_map_gv.rigs)

    # load pairsfile
    pairs = {}
    with open(pairsfile_path, 'r') as fid:
        table = table_from_file(fid)
        for img_query, img_map, score in table:
            if not img_query in pairs:
                pairs[img_query] = []
            pairs[img_query].append(img_map)

    kdata_sub_colmap_path = os.path.join(output_path_root, 'colmap')
    kdata_reg_query_path = os.path.join(output_path_root, 'query_registered')
    sub_kapture_pairsfile_path = os.path.join(output_path_root, 'tmp_pairs_map.txt')
    query_img_kapture_pairsfile_path = os.path.join(output_path_root, 'tmp_pairs_query.txt')

    # loop over query images
    for img_query, img_list_map in pairs.items():
        if pose_found(kdata_query, img_query):
            logger.info(f'{img_query} already processed, skipping...')
            continue
        else:
            logger.info(f'processing {img_query}')

        # write pairsfile for sub-kapture
        map_pairs = write_pairfile_from_img_list(img_list_map, sub_kapture_pairsfile_path)

        # write pairsfile for query_img_kapture
        query_pairs = write_pairfile_img_vs_img_list(img_query, img_list_map, query_img_kapture_pairsfile_path)

        # create sub-kapture
        kdata_sub = sub_kapture_from_img_list(kdata_map, map_plus_query_path, img_list_map, map_pairs)
        kdata_sub_gv = sub_kapture_from_img_list(kdata_map_gv, map_plus_query_gv_path, img_list_map, map_pairs)

        # match missing pairs for mapping
        compute_matches_from_loaded_data(map_plus_query_path, kdata_sub, map_pairs)

        # kdata_sub needs to be re-created to add the new matches
        kdata_sub = sub_kapture_from_img_list(kdata_map, map_plus_query_path, img_list_map, map_pairs)

        # run colmap gv on missing pairs
        if len(kdata_sub.matches) != len(kdata_sub_gv.matches):
            run_colmap_gv_from_loaded_data(kdata_sub,
                                           kdata_sub_gv,
                                           map_plus_query_path,
                                           map_plus_query_gv_path,
                                           colmap_binary,
                                           [],
                                           True)
            # kdata_sub_gv needs to be re-created to add the new matches
            kdata_sub_gv = sub_kapture_from_img_list(kdata_map_gv, map_plus_query_gv_path, img_list_map, map_pairs)

        # sanity check
        if len(map_pairs) != len(kdata_sub_gv.matches):
            logger.info(f'not all mapping matches available')

        # build COLMAP map
        try:
            colmap_build_map_from_loaded_data(
                kdata_sub_gv,
                map_plus_query_gv_path,
                kdata_sub_colmap_path,
                colmap_binary,
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

        # create single image kapture (kdata_sub needs to be recreated because descriptors are deleted in build_colmap_model)
        kdata_sub = sub_kapture_from_img_list(kdata_map, map_plus_query_path, img_list_map, map_pairs)
        kdata_sub_gv = sub_kapture_from_img_list(kdata_map_gv, map_plus_query_gv_path, img_list_map, map_pairs)
        query_img_kapture = add_image_to_kapture(kdata_map, map_plus_query_path, kdata_sub, img_query, query_pairs)
        query_img_kapture_gv = add_image_to_kapture(kdata_map_gv, map_plus_query_gv_path, kdata_sub_gv, img_query,
                                                    query_pairs)

        # match missing pairs for localization
        compute_matches_from_loaded_data(map_plus_query_path, query_img_kapture, query_pairs)

        # query_img_kapture needs to be re-created to add the new matches
        query_img_kapture = add_image_to_kapture(kdata_map, map_plus_query_path, kdata_sub, img_query, query_pairs)

        # run colmap gv on missing pairs
        if len(query_img_kapture.matches) != len(query_img_kapture_gv.matches):
            run_colmap_gv_from_loaded_data(query_img_kapture,
                                           query_img_kapture_gv,
                                           map_plus_query_path,
                                           map_plus_query_gv_path,
                                           colmap_binary,
                                           [],
                                           True)
            # query_img_kapture_gv needs to be re-created to add the new matches
            query_img_kapture_gv = add_image_to_kapture(kdata_map_gv, map_plus_query_gv_path, kdata_sub_gv, img_query,
                                                        query_pairs)

        # sanity check
        if len(query_pairs) != len(query_img_kapture_gv.matches):
            logger.info(f'not all query matches available')

        # localize in COLMAP map
        try:
            colmap_localize_from_loaded_data(
                query_img_kapture_gv,
                map_plus_query_gv_path,
                os.path.join(kdata_sub_colmap_path, 'registered'),
                os.path.join(kdata_sub_colmap_path, 'colmap.db'),
                os.path.join(kdata_sub_colmap_path, 'reconstruction'),
                colmap_binary,
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
    safe_remove_file(query_img_kapture_pairsfile_path, True)

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

    args = parser.parse_args()

    logger.setLevel(args.verbose)
    logging.getLogger('colmap').setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    local_sfm(args.map_plus_query, args.map_plus_query_gv, args.query, args.pairsfile_path, args.output,
              args.colmap_binary, args.force)


if __name__ == '__main__':
    local_sfm_command_line()
