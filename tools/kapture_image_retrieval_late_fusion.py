#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib
from typing import List, Optional

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_to_file
from kapture.io.csv import GlobalFeaturesConfig, kapture_from_dir, table_to_file, get_all_tar_handlers
from kapture.io.tar import TarCollection
from kapture.io.features import get_features_fullpath


import kapture_localization.utils.logging
from kapture_localization.image_retrieval.pairing import stack_global_features, get_image_pairs
from kapture_localization.image_retrieval.pairing import get_similarity_matrix
from kapture_localization.image_retrieval.pairing import get_similarity_dict_from_similarity_matrix
from kapture_localization.image_retrieval.fusion import fuse_similarities, LateFusionMethod
from kapture_localization.image_retrieval.fusion import round_robin_from_similarity_dicts
from kapture_localization.image_retrieval.fusion import get_image_retrieval_late_fusion_argparser
logger = kapture_localization.utils.logging.getLogger()


def image_retrieval_late_fusion(input_path: str,
                                query_path: Optional[str],
                                global_features_types: List[str],
                                output_path: str,
                                topk: Optional[int],
                                method: LateFusionMethod,
                                additional_parameters: dict):
    """
    fuse image retrieval similarities and write a pairsfile with the fused scores
    """
    skip_heavy = [
        kapture.Keypoints, kapture.Descriptors,
        kapture.Matches,
        kapture.Points3d, kapture.Observations
    ]

    logger.info(f'image_retrieval_late_fusion. loading {input_path}')
    with get_all_tar_handlers(input_path) as map_tar_handlers:
        kdata_map = kapture_from_dir(input_path, None, skip_heavy, tar_handlers=map_tar_handlers)
        assert kdata_map.sensors is not None
        assert kdata_map.records_camera is not None
        assert kdata_map.global_features is not None and len(kdata_map.global_features) > 0

        if query_path is None:
            _image_retrieval_late_fusion_from_loaded_data(input_path, map_tar_handlers, kdata_map,
                                                          input_path, map_tar_handlers, kdata_map,
                                                          global_features_types, output_path, topk,
                                                          method, additional_parameters)
        else:
            with get_all_tar_handlers(query_path) as query_tar_handlers:
                logger.info(f'image_retrieval_late_fusion. loading {query_path}')
                kdata_query = kapture_from_dir(query_path, None, skip_heavy, tar_handlers=query_tar_handlers)
                assert kdata_query.sensors is not None
                assert kdata_query.records_camera is not None
                assert kdata_query.global_features is not None and len(kdata_query.global_features) > 0
                _image_retrieval_late_fusion_from_loaded_data(input_path, map_tar_handlers, kdata_map,
                                                              query_path, query_tar_handlers, kdata_query,
                                                              global_features_types, output_path, topk,
                                                              method, additional_parameters)


def _image_retrieval_late_fusion_from_loaded_data(input_path: str,
                                                  map_tar_handlers: TarCollection,
                                                  kdata_map: kapture.Kapture,
                                                  query_path: str,
                                                  query_tar_handlers: TarCollection,
                                                  kdata_query: kapture.Kapture,
                                                  global_features_types: List[str],
                                                  output_path: str,
                                                  topk: Optional[int],
                                                  method: LateFusionMethod,
                                                  additional_parameters: dict):
    image_list_map = [name for _, _, name in kapture.flatten(kdata_map.records_camera, is_sorted=True)]
    image_list_query = [name for _, _, name in kapture.flatten(kdata_query.records_camera, is_sorted=True)]

    if len(global_features_types) == 0:
        global_features_types = list(set(kdata_map.global_features.keys()
                                         ).intersection(
                                             kdata_query.global_features.keys()
        ))

    similarity_matrices = []
    stacked_query_index = None
    stacked_map_index = None

    for global_features_type in global_features_types:
        if global_features_type not in kdata_map.global_features:
            logger.warning(f'could not use {global_features_type}, it was missing in kdata_map')
            continue
        if global_features_type not in kdata_query.global_features:
            logger.warning(f'could not use {global_features_type}, it was missing in kdata_query')
            continue
        mapping_gfeats = kdata_map.global_features[global_features_type]
        query_gfeats = kdata_query.global_features[global_features_type]
        assert mapping_gfeats.dtype == query_gfeats.dtype
        assert mapping_gfeats.dsize == query_gfeats.dsize
        assert mapping_gfeats.metric_type == query_gfeats.metric_type

        global_features_config = GlobalFeaturesConfig(mapping_gfeats.type_name,
                                                      mapping_gfeats.dtype,
                                                      mapping_gfeats.dsize,
                                                      mapping_gfeats.metric_type)

        # force the same order for all global features
        mapping_global_features_to_filepaths = [
            (image_filename, get_features_fullpath(kapture.GlobalFeatures, global_features_type,
                                                   input_path, image_filename, map_tar_handlers))
            for image_filename in image_list_map
        ]
        mapping_stacked_features = stack_global_features(global_features_config, mapping_global_features_to_filepaths)

        if input_path == query_path:
            query_stacked_features = mapping_stacked_features
        else:
            query_global_features_to_filepaths = [
                (image_filename, get_features_fullpath(kapture.GlobalFeatures, global_features_type,
                                                       query_path, image_filename, query_tar_handlers))
                for image_filename in image_list_query
            ]
            query_stacked_features = stack_global_features(global_features_config, query_global_features_to_filepaths)

        # additional step to really make sure the order or the matrix is the same, and to remember it
        if stacked_map_index is None:
            stacked_map_index = mapping_stacked_features.index
        else:
            assert stacked_map_index.tolist() == mapping_stacked_features.index.tolist()

        if stacked_query_index is None:
            stacked_query_index = query_stacked_features.index
        else:
            assert stacked_query_index.tolist() == query_stacked_features.index.tolist()

        similarity_matrices.append(get_similarity_matrix(query_stacked_features, mapping_stacked_features))

    if method == LateFusionMethod.round_robin:
        logger.info(f'Compute fused similarity from round_robin')
        similarity_dicts = [get_similarity_dict_from_similarity_matrix(similarity,
                                                                       stacked_query_index,
                                                                       stacked_map_index)
                            for similarity in similarity_matrices]
        image_pairs = round_robin_from_similarity_dicts(similarity_dicts, topk)
    else:
        logger.info(f'Compute fused similarity from {method.value} ...')
        similarity = fuse_similarities(similarity_matrices, method, additional_parameters)
        similarity_dict = get_similarity_dict_from_similarity_matrix(similarity, stacked_query_index, stacked_map_index)
        image_pairs = get_image_pairs(similarity_dict, topk)

    logger.info('saving to file ...')
    os.umask(0o002)
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output_path, 'w') as fid:
        table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    logger.info('all done')


def image_retrieval_late_fusion_command_line():
    parser = argparse.ArgumentParser(
        description='Create image pairs files from fusing multiple global feature similarities.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture input root directory.')
    parser.add_argument('--query', default=None, help='Keep to default (None) when mapping.\n'
                        'input path to a kapture root directory containing query images.\n')
    parser.add_argument('-gfeat', '--global_features_types', nargs='+', default=[],
                        help='Optional; types of the global features to fuse. if not given, they will all be fused')
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

    logger.debug('kapture_image_retrieval_late_fusion.py \\\n' + ''.join(['\n\t{:13} = {}'.format(k, v)
                                                                          for k, v in vars(args).items()]))
    image_retrieval_late_fusion(args.input, args.query, args.global_features_types,
                                args.output, args.topk, args.method, additional_parameters)


if __name__ == '__main__':
    image_retrieval_late_fusion_command_line()
