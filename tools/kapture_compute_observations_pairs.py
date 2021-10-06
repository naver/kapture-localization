#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib
from typing import Optional

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.pairing.observations import get_pairs_observations

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_to_file, get_all_tar_handlers
from kapture.utils.Collections import try_get_only_key_from_collection

logger = logging.getLogger('compute_observations_pairs')


def compute_observations_pairs(mapping_path: str,
                               query_path: Optional[str],
                               output_path: str,
                               topk: int,
                               keypoints_type: Optional[str],
                               iou: bool,
                               max_number_of_threads: Optional[int] = None):
    """
    compute image pairs from observations, and write the result in a text file
    """
    skip_heavy_features = [kapture.Descriptors, kapture.GlobalFeatures, kapture.Matches]
    skip_heavy = [kapture.RecordsLidar, kapture.RecordsWifi] + skip_heavy_features

    logger.info(f'compute_observations_pairs. loading mapping: {mapping_path}')
    # the content of the keypoints is not important, we do not need to keep a reference to the tar
    with get_all_tar_handlers(mapping_path, skip_list=skip_heavy_features) as tar_handlers:
        kdata = kapture_from_dir(mapping_path, skip_list=skip_heavy, tar_handlers=tar_handlers)
    assert kdata.sensors is not None
    assert kdata.records_camera is not None
    if keypoints_type is None:
        keypoints_type = try_get_only_key_from_collection(kdata.keypoints)
    assert keypoints_type is not None
    assert kdata.observations is not None
    assert kdata.keypoints is not None
    assert keypoints_type in kdata.keypoints
    assert kdata.points3d is not None

    if query_path is None or mapping_path == query_path:
        logger.info('computing mapping pairs from observations...')
        kdata_query = None
    else:
        logger.info('computing query pairs from observations...')
        with get_all_tar_handlers(query_path, skip_list=skip_heavy_features) as query_tar_handlers:
            kdata_query = kapture_from_dir(query_path, skip_list=skip_heavy, tar_handlers=query_tar_handlers)
        assert kdata_query.sensors is not None
        assert kdata_query.records_camera is not None

    os.umask(0o002)
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)

    with open(output_path, 'w') as fid:
        image_pairs = get_pairs_observations(kdata, kdata_query, keypoints_type,
                                             max_number_of_threads, iou, topk)
        table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    logger.info('all done')


def compute_observations_pairs_command_line():
    parser = argparse.ArgumentParser(
        description=('Create image pairs files from observations. '
                     'Pairs are computed between query <-> mapping or mapping <-> mapping'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--mapping', required=True, help=('input path to kapture input root directory.'
                                                          ' when query is given, this must be map_plus_query'))
    parser.add_argument('--query', default=None,
                        help=('input path to a kapture root directory containing query images, '
                              'keep to default None when mapping\n'))

    parser.add_argument('--keypoints-type', default=None, help='keypoint type_name')
    parser.add_argument('--iou', action='store_true', default=False,
                        help='use iou')

    parser.add_argument('--max-number-of-threads', default=None, type=int,
                        help='By default, use as many as cpus. But you can set a limit.')

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
    compute_observations_pairs(args.mapping, args.query, args.output, args.topk,
                               args.keypoints_type, args.iou,
                               args.max_number_of_threads)


if __name__ == '__main__':
    compute_observations_pairs_command_line()
