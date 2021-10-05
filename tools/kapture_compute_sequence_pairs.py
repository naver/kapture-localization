#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.pairing.sequence import get_pairs_sequence

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_to_file

logger = logging.getLogger('compute_sequence_pairs')


def compute_sequence_pairs(mapping_path: str,
                           output_path: str,
                           window_size: int,
                           loop: bool,
                           expand_window: bool,
                           max_interval: int):
    """
    compute image pairs from sequence, and write the result in a text file
    """
    skip_heavy = [kapture.RecordsLidar, kapture.RecordsWifi,
                  kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                  kapture.Matches, kapture.Points3d, kapture.Observations]

    logger.info(f'compute_sequence_pairs. loading mapping: {mapping_path}')
    kdata = kapture_from_dir(mapping_path, skip_list=skip_heavy)
    assert kdata.sensors is not None
    assert kdata.records_camera is not None

    os.umask(0o002)
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)

    with open(output_path, 'w') as fid:
        image_pairs = get_pairs_sequence(kdata, window_size, loop, expand_window, max_interval)
        table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    logger.info('all done')


def compute_sequence_pairs_command_line():
    parser = argparse.ArgumentParser(
        description=('Create image pairs files from sequences (same camera). '
                     'Pairs are computed between mapping <-> mapping only'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--mapping', required=True,
                        help='input path to kapture input root directory')

    parser_dist = parser.add_mutually_exclusive_group()
    parser_dist.add_argument('--loop', action='store_true', default=False, help='all the sequences loop')
    parser_dist.add_argument('--expand-window', action='store_true', default=False,
                             help='expand window right or left when near the edge of the array')

    parser.add_argument('--max-interval', type=int, default=5000000,
                        help='Maximum time interval between records to be considered within the same sequence')

    parser.add_argument('-o', '--output', required=True,
                        help='output path to pairsfile')

    parser.add_argument('--window-size',
                        default=10, type=int,
                        help='match with n images forward, and n images backward when possible')

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_sequence_pairs(args.mapping, args.output, args.window_size,
                           args.loop, args.expand_window, args.max_interval)


if __name__ == '__main__':
    compute_sequence_pairs_command_line()
