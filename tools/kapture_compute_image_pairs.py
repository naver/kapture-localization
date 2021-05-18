#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib
from typing import Optional

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.image_retrieval.pairing import stack_global_features, get_similarity, get_image_pairs

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_to_file
from kapture.io.csv import GlobalFeaturesConfig, get_all_tar_handlers
from kapture.io.features import global_features_to_filepaths
from kapture.utils.Collections import try_get_only_key_from_collection

logger = logging.getLogger('compute_image_pairs')


def compute_image_pairs(mapping_path: str,
                        query_path: str,
                        output_path: str,
                        global_features_type: Optional[str],
                        topk: int):
    """
    compute image pairs between query -> mapping from global features, and write the result in a text file

    :param mapping_path: input path to kapture input root directory
    :type mapping_path: str
    :param query_path: input path to a kapture root directory
    :type query_path: str
    :param output_path: output path to pairsfile
    :type output_path: str
    :param global_features_type: type of global_features, name of the global_features subfolder
    :param topk: the max number of top retained images
    :type topk: int
    """
    logger.info(f'compute_image_pairs. loading mapping: {mapping_path}')
    with get_all_tar_handlers(mapping_path) as mapping_tar_handlers:
        kdata_mapping = kapture_from_dir(mapping_path, None, skip_list=[kapture.Keypoints,
                                                                        kapture.Descriptors,
                                                                        kapture.Matches,
                                                                        kapture.Observations,
                                                                        kapture.Points3d],
                                         tar_handlers=mapping_tar_handlers)
        assert kdata_mapping.sensors is not None
        assert kdata_mapping.records_camera is not None
        assert kdata_mapping.global_features is not None
        if global_features_type is None:
            global_features_type = try_get_only_key_from_collection(kdata_mapping.global_features)
        assert global_features_type is not None
        assert global_features_type in kdata_mapping.global_features

        global_features_config = GlobalFeaturesConfig(kdata_mapping.global_features[global_features_type].type_name,
                                                      kdata_mapping.global_features[global_features_type].dtype,
                                                      kdata_mapping.global_features[global_features_type].dsize,
                                                      kdata_mapping.global_features[global_features_type].metric_type)

        logger.info(f'computing pairs with {global_features_type}...')

        mapping_global_features_to_filepaths = global_features_to_filepaths(
            kdata_mapping.global_features[global_features_type],
            global_features_type,
            mapping_path,
            mapping_tar_handlers
        )
        mapping_list = list(sorted(mapping_global_features_to_filepaths.items()))
        mapping_stacked_features = stack_global_features(global_features_config, mapping_list)

    if mapping_path == query_path:
        kdata_query = kdata_mapping
        query_stacked_features = mapping_stacked_features
    else:
        logger.info(f'compute_image_pairs. loading query: {query_path}')
        with get_all_tar_handlers(query_path) as query_tar_handlers:
            kdata_query = kapture_from_dir(query_path, None, skip_list=[kapture.Keypoints,
                                                                        kapture.Descriptors,
                                                                        kapture.Matches,
                                                                        kapture.Observations,
                                                                        kapture.Points3d],
                                           tar_handlers=query_tar_handlers)
            assert kdata_query.sensors is not None
            assert kdata_query.records_camera is not None
            assert kdata_query.global_features is not None
            assert global_features_type in kdata_query.global_features

            kdata_mapping_gfeat = kdata_mapping.global_features[global_features_type]
            kdata_query_gfeat = kdata_query.global_features[global_features_type]
            assert kdata_mapping_gfeat.type_name == kdata_query_gfeat.type_name
            assert kdata_mapping_gfeat.dtype == kdata_query_gfeat.dtype
            assert kdata_mapping_gfeat.dsize == kdata_query_gfeat.dsize

            query_global_features_to_filepaths = global_features_to_filepaths(
                kdata_query_gfeat,
                global_features_type,
                query_path,
                query_tar_handlers
            )
            query_list = list(sorted(query_global_features_to_filepaths.items()))
            query_stacked_features = stack_global_features(global_features_config, query_list)

    similarity = get_similarity(query_stacked_features, mapping_stacked_features)

    # get list of image pairs
    image_pairs = get_image_pairs(similarity, topk)

    logger.info('saving to file  ...')
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output_path, 'w') as fid:
        table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    logger.info('all done')


def compute_image_pairs_command_line():
    parser = argparse.ArgumentParser(
        description=('Create image pairs files from global features. '
                     'Pairs are computed between query <-> mapping or mapping <-> mapping'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--mapping', required=True,
                        help=('input path to kapture input root directory\n'
                              'it must contain global features for all images'))
    parser.add_argument('--query', required=True,
                        help=('input path to a kapture root directory containing query images\n'
                              'it must contain global features for all images\n'
                              'use the same value as mapping if you want to compute mapping <-> mapping matches'))
    parser.add_argument('-o', '--output', required=True,
                        help='output path to pairsfile')
    parser.add_argument('-gfeat', '--global-features-type', default=None, help='kapture global features type.')
    parser.add_argument('--topk',
                        default=20,
                        type=int,
                        help='the max number of top retained images')

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_image_pairs(args.mapping, args.query, args.output, args.global_features_type, args.topk)


if __name__ == '__main__':
    compute_image_pairs_command_line()
