#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib
from typing import Optional
import math
from tqdm import tqdm

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.pairing.distance import get_pairs_distance

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_to_file

logger = logging.getLogger('compute_distance_pairs')


def compute_distance_pairs(mapping_path: str,
                           query_path: Optional[str],
                           output_path: str,
                           topk: int,
                           block_size: int,
                           min_distance: float,
                           max_distance: float,
                           max_angle: float,
                           keep_rejected: bool):
    """
    compute image pairs from distance, and write the result in a text file
    """
    skip_heavy = [kapture.RecordsLidar, kapture.RecordsWifi,
                  kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                  kapture.Matches, kapture.Points3d, kapture.Observations]

    logger.info(f'compute_distance_pairs. loading mapping: {mapping_path}')
    kdata = kapture_from_dir(mapping_path, skip_list=skip_heavy)
    assert kdata.sensors is not None
    assert kdata.records_camera is not None
    assert kdata.trajectories is not None

    if query_path is None or mapping_path == query_path:
        logger.info('computing mapping pairs from distance...')
        kdata_query = None
    else:
        logger.info('computing query pairs from distance...')
        kdata_query = kapture_from_dir(query_path, skip_list=skip_heavy)
        assert kdata_query.sensors is not None
        assert kdata_query.records_camera is not None
        assert kdata_query.trajectories is not None

    os.umask(0o002)
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)

    with open(output_path, 'w') as fid:
        if kdata_query is None:
            kdata_query = kdata
        if kdata_query.rigs is not None:
            assert kdata_query.trajectories is not None  # for ide
            kapture.rigs_remove_inplace(kdata_query.trajectories, kdata_query.rigs)
        records_camera_list = [k
                               for k in sorted(kapture.flatten(kdata_query.records_camera),
                                               key=lambda x: x[2])]
        number_of_iteration = math.ceil(len(records_camera_list) / block_size)
        table_to_file(fid, [], header='# query_image, map_image, score')
        for i in tqdm(range(number_of_iteration), disable=logging.getLogger().level >= logging.CRITICAL):
            sliced_records = kapture.RecordsCamera()
            for ts, sensor_id, img_name in records_camera_list[i * block_size:(i+1)*block_size]:
                if (ts, sensor_id) not in kdata_query.trajectories:
                    continue
                sliced_records[(ts, sensor_id)] = img_name
            kdata_slice_query = kapture.Kapture(
                sensors=kdata_query.sensors,
                records_camera=sliced_records,
                trajectories=kdata_query.trajectories
            )
            image_pairs = get_pairs_distance(kdata, kdata_slice_query, topk,
                                             min_distance, max_distance, max_angle,
                                             keep_rejected)
            table_to_file(fid, image_pairs)
    logger.info('all done')


def compute_distance_pairs_command_line():
    parser = argparse.ArgumentParser(
        description=('Create image pairs files from distance. Does not handle none poses'
                     'Pairs are computed between query <-> mapping or mapping <-> mapping'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--mapping', required=True, help='input path to kapture input root directory')
    parser.add_argument('--query', default=None,
                        help=('input path to a kapture root directory containing query images, '
                              'keep to default None when mapping\n'))

    parser.add_argument('--max-distance', type=float, default=25.0,
                        help='max distance to form a pair')
    parser.add_argument('--max-angle', type=float, default=45.0,
                        help='max angle to form a pair')

    parser_dist = parser.add_mutually_exclusive_group()
    parser_dist.add_argument('--min-distance', type=float, default=0.0,
                             help='min distance to form a pair')
    parser_dist.add_argument('--keep-rejected', action='store_true', default=False,
                             help='keep pairs that are not within the thresholds bounds')

    parser.add_argument('--block-size', default=1000, type=int,
                        help=('number of (query) images to process at once'))

    parser.add_argument('-o', '--output', required=True,
                        help='output path to pairsfile')
    parser.add_argument('--topk', default=None, type=int,
                        help='the max number of top retained images')

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_distance_pairs(args.mapping, args.query, args.output, args.topk,
                           args.block_size,
                           args.min_distance, args.max_distance, args.max_angle,
                           args.keep_rejected)


if __name__ == '__main__':
    compute_distance_pairs_command_line()
