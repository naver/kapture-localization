#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import os
import logging
from typing import Optional
import numpy as np

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
from kapture.io.structure import delete_existing_kapture_files
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, kapture_to_dir
from kapture.algo.pose_operations import average_pose_transform_weighted

logger = logging.getLogger('pose_approximation')

METHOD_DESCRIPTIONS = {
    'equal_weighted_barycenter': ("EWB: assigns the same weight to all of the top k retrieved "
                                  "images with w_i = 1/k"),
    'cosine_similarity': ("CSI: w_i=(1/z_i)*(transpose(d_q)*d_i)^alpha, "
                          "z_i=sum(transpose(d_q)*d_j)^alpha")
}


def pose_approximation_from_pairsfile(input_path: str,
                                      pairsfile_path: str,
                                      output_path: str,
                                      query_path: Optional[str],
                                      topk: Optional[int],
                                      method: str,
                                      additional_parameters: dict,
                                      force: bool):
    """
    localize from pairsfile
    """
    os.makedirs(output_path, exist_ok=True)
    delete_existing_kapture_files(output_path, force_erase=force)

    logger.info(f'pose_approximation. loading mapping: {input_path}')
    kdata = kapture_from_dir(input_path, None, skip_list=[kapture.Keypoints,
                                                          kapture.Descriptors,
                                                          kapture.GlobalFeatures,
                                                          kapture.Matches,
                                                          kapture.Points3d,
                                                          kapture.Observations])
    if query_path is not None:
        logger.info(f'pose_approximation. loading query: {query_path}')
        kdata_query = kapture_from_dir(query_path, skip_list=[kapture.Keypoints,
                                                              kapture.Descriptors,
                                                              kapture.GlobalFeatures,
                                                              kapture.Matches,
                                                              kapture.Points3d,
                                                              kapture.Observations])
    else:
        kdata_query = kdata

    logger.info(f'pose_approximation. loading pairs: {pairsfile_path}')
    similarity_dict = get_ordered_pairs_from_file(pairsfile_path, kdata_query.records_camera,
                                                  kdata.records_camera, topk)
    query_images = set(similarity_dict.keys())

    kdata_result = kapture.Kapture(sensors=kapture.Sensors(),
                                   records_camera=kapture.RecordsCamera(),
                                   trajectories=kapture.Trajectories())
    for timestamp, cam_id, image_name in kapture.flatten(kdata_query.records_camera):
        if image_name not in query_images:
            continue
        if cam_id not in kdata_result.sensors:
            kdata_result.sensors[cam_id] = kdata_query.sensors[cam_id]
        kdata_result.records_camera[(timestamp, cam_id)] = image_name

    if kdata.rigs is None:
        map_trajectories = kdata.trajectories
    else:
        map_trajectories = kapture.rigs_remove(kdata.trajectories, kdata.rigs)
    training_trajectories_reversed = {image_name: map_trajectories[(timestamp, cam_id)]
                                      for timestamp, cam_id, image_name in kapture.flatten(kdata.records_camera)
                                      if (timestamp, cam_id) in map_trajectories}
    records_camera_reversed = {image_name: (timestamp, cam_id)
                               for timestamp, cam_id, image_name in kapture.flatten(kdata_result.records_camera)}

    for image_name, similar_images in similarity_dict.items():
        pose_inv_list = [training_trajectories_reversed[k].inverse() for k, _ in similar_images]
        timestamp = records_camera_reversed[image_name][0]
        cam_id = records_camera_reversed[image_name][1]

        if method == 'equal_weighted_barycenter':
            weight_list = [1.0/len(pose_inv_list) for _ in range(len(pose_inv_list))]
        else:
            assert 'alpha' in additional_parameters
            alpha = additional_parameters['alpha']
            weights = np.zeros((len(pose_inv_list),))
            for i, (_, score) in enumerate(similar_images):
                weights[i] = score
            weights[:] = weights[:]**(alpha)
            weights[:] = weights[:] / np.sum(weights[:])
            weight_list = weights.tolist()
        final_pose = average_pose_transform_weighted(pose_inv_list, weight_list).inverse()
        kdata_result.trajectories[(timestamp, cam_id)] = final_pose

    kapture_to_dir(output_path, kdata_result)
    logger.info('all done')


def get_pose_approximation_method_argparser(method: str):
    parser_method = argparse.ArgumentParser(description=METHOD_DESCRIPTIONS[method],
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_method.set_defaults(method=method)
    # per method parameters
    if method == 'cosine_similarity':
        parser_method.add_argument('--alpha', default=8.0, type=float, help='alpha parameter of CSI')
    return parser_method


def pose_approximation_from_pairsfile_command_line():
    parser = argparse.ArgumentParser(description='localize from pairfile',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--mapping', required=True,
                        help=('input path to kapture input root directory.\n'
                              'if query is left to None, it must contains all images'))
    parser.add_argument('-o', '--output', required=True, help='output path to localized queries')
    parser.add_argument('--query', default=None,
                        help='if left to None, timestamp, sensor_id will be taken from input, else from this')
    parser.add_argument('--pairsfile-path', required=True, type=str,
                        help='text file which contains the image pairs and their score')
    parser.add_argument('--topk', default=None, type=int,
                        help='override pairfile topk with this one (must be inferior or equal)')
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output directory if already exists')
    list_of_pose_approx_methods = ['equal_weighted_barycenter', 'cosine_similarity']
    valid_subcommands = ', '.join(list_of_pose_approx_methods)
    subparsers = parser.add_subparsers(title='subcommands',
                                       description=f'valid subcommands: {valid_subcommands}',
                                       help='additional help')
    for method in list_of_pose_approx_methods:
        subparsers.choices[method] = get_pose_approximation_method_argparser(method)
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug('pose_approximation_from_pairsfile.py \\\n' + ''.join(['\n\t{:13} = {}'.format(k, v)
                                                                        for k, v in vars(args).items()]))
    pose_approximation_from_pairsfile(args.mapping, args.pairsfile_path, args.output,
                                      args.query, args.topk,
                                      args.method, vars(args),
                                      args.force)


if __name__ == '__main__':
    pose_approximation_from_pairsfile_command_line()
