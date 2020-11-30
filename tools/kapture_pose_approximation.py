#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.image_retrieval.pairing import stack_global_features
from kapture_localization.pose_approximation import PoseApproximationMethods, METHOD_DESCRIPTIONS
from kapture_localization.pose_approximation.weight_estimation import get_interpolation_weights
from kapture_localization.pose_approximation.pose_interpolation import get_interpolated_pose

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, ImageFeatureConfig, kapture_to_dir
from kapture.io.features import global_features_to_filepaths
from kapture.io.structure import delete_existing_kapture_files

logger = logging.getLogger('pose_approximation')


def pose_approximation(mapping_path: str,
                       query_path: str,
                       output_path: str,
                       topk: int,
                       force_overwrite_existing: bool,
                       method: PoseApproximationMethods,
                       additional_parameters: dict):
    """
    compute approximated pose from image retrieval results

    :param mapping_path: input path to kapture input root directory
    :type mapping_path: str
    :param query_path: input path to a kapture root directory
    :type query_path: str
    :param output_path: output path to pairsfile
    :type output_path: str
    :param topk: the max number of top retained images
    :type topk: int
    :param additional_parameters: store method specific args
    :type additional_parameters: dict
    """
    assert mapping_path != query_path

    os.makedirs(output_path, exist_ok=True)
    delete_existing_kapture_files(output_path, force_erase=force_overwrite_existing)

    logger.info(f'pose_approximation. loading mapping: {mapping_path}')
    kdata_map = kapture_from_dir(mapping_path, None, skip_list=[kapture.Keypoints,
                                                                kapture.Descriptors,
                                                                kapture.Matches,
                                                                kapture.Observations,
                                                                kapture.Points3d])
    assert kdata_map.sensors is not None
    assert kdata_map.records_camera is not None
    assert kdata_map.global_features is not None

    logger.info(f'pose_approximation. loading query: {query_path}')
    kdata_query = kapture_from_dir(query_path, None, skip_list=[kapture.Keypoints,
                                                                kapture.Descriptors,
                                                                kapture.Matches,
                                                                kapture.Observations,
                                                                kapture.Points3d])
    assert kdata_query.sensors is not None
    assert kdata_query.records_camera is not None
    assert kdata_query.global_features is not None

    assert kdata_map.global_features is not None
    assert kdata_query.global_features is not None
    assert kdata_map.global_features.type_name == kdata_query.global_features.type_name
    assert kdata_map.global_features.dtype == kdata_query.global_features.dtype
    assert kdata_map.global_features.dsize == kdata_query.global_features.dsize
    global_features_config = ImageFeatureConfig(kdata_map.global_features.type_name,
                                                kdata_map.global_features.dtype,
                                                kdata_map.global_features.dsize)

    logger.info(f'computing pose approximation from with {kdata_map.global_features.type_name}...')

    map_global_features_to_filepaths = global_features_to_filepaths(kdata_map.global_features,
                                                                    mapping_path)
    mapping_list = list(kapture.flatten(map_global_features_to_filepaths, is_sorted=True))
    map_stacked_features = stack_global_features(global_features_config, mapping_list)

    query_global_features_to_filepaths = global_features_to_filepaths(kdata_query.global_features,
                                                                      query_path)
    query_list = list(kapture.flatten(query_global_features_to_filepaths, is_sorted=True))
    query_stacked_features = stack_global_features(global_features_config, query_list)

    # main code
    weights = get_interpolation_weights(method,
                                        query_stacked_features,
                                        map_stacked_features,
                                        topk,
                                        additional_parameters)
    out_trajectories = get_interpolated_pose(kdata_map, kdata_query, weights)
    out_kapture = kapture.Kapture(sensors=kdata_query.sensors,
                                  records_camera=kdata_query.records_camera,
                                  trajectories=out_trajectories)
    kapture_to_dir(output_path, out_kapture)
    logger.info('all done')


def get_pose_approximation_method_argparser(method: PoseApproximationMethods):
    parser_method = argparse.ArgumentParser(description=METHOD_DESCRIPTIONS[method],
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_method.set_defaults(method=method)
    # per method parameters
    if method == PoseApproximationMethods.cosine_similarity:
        parser_method.add_argument('--alpha', default=8.0, type=float, help='alpha parameter of CSI')
    return parser_method


def pose_approximation_command_line():
    parser = argparse.ArgumentParser(
        description=('compute approximated pose from image retrieval results'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete colmap if already exists.')
    parser.add_argument('--mapping', required=True,
                        help=('input path to kapture input root directory\n'
                              'it must contain global features for all images'))
    parser.add_argument('--query', required=True,
                        help=('input path to a kapture root directory containing query images\n'
                              'it must contain global features for all images\n'
                              'use the same value as mapping if you want to compute mapping <-> mapping matches'))
    parser.add_argument('-o', '--output', required=True, help='output folder for estimated query trajectories')
    parser.add_argument('--topk',
                        default=20,
                        type=int,
                        help='the max number of top retained images')

    list_of_pose_approx_methods = list(PoseApproximationMethods)
    valid_subcommands = ', '.join([method.value for method in list_of_pose_approx_methods])
    subparsers = parser.add_subparsers(title='subcommands',
                                       description=f'valid subcommands: {valid_subcommands}',
                                       help='additional help')
    for method in list_of_pose_approx_methods:
        subparsers.choices[method.value] = get_pose_approximation_method_argparser(method)

    args = parser.parse_args()

    # store method specific args in this dict
    additional_parameters = vars(args)

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    pose_approximation(args.mapping, args.query, args.output, args.topk, args.force,
                       args.method, additional_parameters)


if __name__ == '__main__':
    pose_approximation_command_line()
