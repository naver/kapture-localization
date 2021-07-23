#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib
from typing import List, Optional
import numpy as np

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import table_to_file, table_to_file

import kapture_localization.utils.logging
from kapture_localization.image_retrieval.pairing import get_image_pairs
from kapture_localization.image_retrieval.fusion import fuse_similarities, LateFusionMethod
from kapture_localization.image_retrieval.fusion import round_robin_from_similarity_dicts
from kapture_localization.image_retrieval.fusion import get_image_retrieval_late_fusion_argparser
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file

logger = kapture_localization.utils.logging.getLogger()


def pairsfile_fusion(input_path: List[str],
                     output_path: str,
                     topk: Optional[int],
                     method: LateFusionMethod,
                     additional_parameters: dict):
    """
    fuse pairsfile scores and write a pairsfile with the fused scores
    """
    assert len(input_path) > 1

    logger.info(f'pairsfile_fusion. loading {input_path}')

    similarity_dicts: List[dict] = []
    for file_path in input_path:
        loaded_pairs = get_ordered_pairs_from_file(file_path)
        similarity_dicts.append(loaded_pairs)

    if method == LateFusionMethod.round_robin:
        image_pairs = round_robin_from_similarity_dicts(similarity_dicts, topk)
    else:
        pairs = {}
        for loaded_pairs in similarity_dicts:
            for query_name, pairlist in loaded_pairs.items():
                for map_name, score in pairlist:
                    pairtuple = (query_name, map_name)
                    if pairtuple not in pairs:
                        pairs[pairtuple] = []
                    pairs[pairtuple].append(score)

        # keep entries with correct count
        similarity_dict = {}
        for pairtuple, scores in pairs.items():
            (query_name, map_name) = pairtuple
            if len(scores) != len(similarity_dicts):
                logger.warning(f'pair {pairtuple} did not have a line in all pairsfile, skipped')
                continue
            scores_as_matrices = [np.array([[score]], dtype=np.float64) for score in scores]
            final_score = fuse_similarities(scores_as_matrices, method, additional_parameters)[0, 0]
            if query_name not in similarity_dict:
                similarity_dict[query_name] = []
            similarity_dict[query_name].append((map_name, final_score))
        image_pairs = get_image_pairs(similarity_dict, topk)

    logger.info('saving to file ...')
    os.umask(0o002)
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output_path, 'w') as fid:
        table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    logger.info('all done')


def pairsfile_fusion_command_line():
    parser = argparse.ArgumentParser(
        description='Create image pairs files from fusing multiple global feature similarities.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', nargs='+', default=[], help='pairsfile inputs')
    parser.add_argument('-o', '--output', required=True, help='output path to pairfile')
    parser.add_argument('--topk', default=None, type=int,
                        help='the max number of top retained images')
    list_of_fusion_methods = list(LateFusionMethod)
    valid_subcommands = ', '.join([method.value for method in list_of_fusion_methods])
    subparsers = parser.add_subparsers(title='subcommands',
                                       description=f'valid subcommands: {valid_subcommands}',
                                       help='additional help')
    for method in list_of_fusion_methods:
        subparsers.choices[method.value] = get_image_retrieval_late_fusion_argparser(method)

    args = parser.parse_args()
    additional_parameters = vars(args)
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug('kapture_pairsfile_fusion.py \\\n' + ''.join(['\n\t{:13} = {}'.format(k, v)
                                                               for k, v in vars(args).items()]))
    pairsfile_fusion(args.input, args.output, args.topk, args.method, additional_parameters)


if __name__ == '__main__':
    pairsfile_fusion_command_line()
