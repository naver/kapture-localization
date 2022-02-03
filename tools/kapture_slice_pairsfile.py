#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import os
import logging
import pathlib
import math

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import table_to_file

logger = kapture_localization.utils.logging.getLogger()


def slice_pairsfile(pairsfile_path: str,
                    output_path: str,
                    topk: int,
                    threshold: float,
                    startk: int,
                    skip_if_na: bool):
    logger.info('slice_pairsfile...')
    similarity_dict = get_ordered_pairs_from_file(pairsfile_path)

    # apply topk override + skip_if_na
    image_pairs = []
    for name_query, paired_images in sorted(similarity_dict.items()):
        paired_images_threshold = [x for x in paired_images if x[1] >= threshold]
        if math.isfinite(topk) and startk + topk > len(paired_images_threshold):
            logger.debug(
                f'image {name_query} has {len(paired_images_threshold)} pairs, '
                f'less than topk={topk} (with startk={startk})')
            if skip_if_na:
                logger.debug(f'skipping {name_query}')
                continue
        if math.isinf(topk):
            paired_images_threshold = paired_images_threshold[startk:]
        else:
            paired_images_threshold = paired_images_threshold[startk:startk+topk]
        for name_map, score in paired_images_threshold:
            image_pairs.append((name_query, name_map, score))

    if len(image_pairs) > 0:
        os.umask(0o002)
        p = pathlib.Path(output_path)
        os.makedirs(str(p.parent.resolve()), exist_ok=True)
        with open(output_path, 'w') as fid:
            table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    else:
        logger.info('no pairs written')
    logger.info('all done')


def slice_pairsfile_command_line():
    parser = argparse.ArgumentParser(description='Apply topk override / threshold on a pairsfile',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True, help='path to input pairsfile')
    parser.add_argument('-o', '--output', required=True, help='path to output pairsfile')
    parser.add_argument('--topk',
                        default=float('inf'),
                        type=int,
                        help='override pairfile topk with this one (must be inferior or equal)')
    parser.add_argument('--threshold', type=float, default=0,
                        help='the minimum score threshold for pairs to be used')
    parser.add_argument('--startk',
                        default=0,
                        type=int,
                        help='start position of topk')
    parser.add_argument('--skip-if-na', action='store_true', default=False,
                        help=('Skip query image if startk + topk greater than available pairs (i.e. na, not available)'
                              '; does not apply when topk is infinity'))
    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug('kapture_slice_pairsfile.py \\\n' + ''.join(['\n\t{:13} = {}'.format(k, v)
                                                              for k, v in vars(args).items()]))
    slice_pairsfile(args.input, args.output, args.topk, args.threshold, args.startk, args.skip_if_na)


if __name__ == '__main__':
    slice_pairsfile_command_line()
