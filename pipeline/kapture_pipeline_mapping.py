#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script builds a COLMAP model (map) from kapture format (images, cameras, trajectories, features!)
"""

import argparse
import logging
import os
import os.path as path
import sys
from typing import List, Optional

import pipeline_import_paths  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.symlink import can_use_symlinks, create_kapture_proxy
from kapture_localization.utils.subprocess import run_python_command

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging
from kapture.utils.paths import safe_remove_file

logger = logging.getLogger('mapping_pipeline')


def mapping_pipeline(kapture_path: str,
                     keypoints_path: str,
                     descriptors_path: str,
                     global_features_path: str,
                     matches_path: str,
                     matches_gv_path: str,
                     colmap_map_path: str,
                     colmap_binary: str,
                     python_binary: Optional[str],
                     topk: int,
                     skip_list: List[str],
                     force_overwrite_existing: bool) -> None:
    """
    Build a colmap model using pre computed features with the kapture data.

    :param kapture_path: path to the kapture map directory
    :type kapture_path: str
    :param keypoints_path: input path to the orphan keypoints folder
    :type keypoints_path: str
    :param descriptors_path: input path to the orphan descriptors folder
    :type descriptors_path: str
    :param global_features_path: input path to the orphan global_features folder
    :type global_features_path: str
    :param matches_path: input path to the orphan matches (not verified) folder
    :type matches_path: str
    :param matches_gv_path: input path to the orphan matches (verified) folder
    :type matches_gv_path: str
    :param colmap_map_path: path to the colmap output folder
    :type colmap_map_path: str
    :param colmap_binary: path to the colmap executable
    :type colmap_binary: str
    :param python_binary: path to the python executable
    :type python_binary: Optional[str]
    :param topk: the max number of top retained images when computing image pairs from global features
    :type topk: int
    :param skip_list: list of steps to ignore
    :type skip_list: List[str]
    :param force_overwrite_existing: silently overwrite files if already exists
    :type force_overwrite_existing: bool
    """
    os.makedirs(colmap_map_path, exist_ok=True)
    pairfile_path = path.join(colmap_map_path, f'pairs_mapping_{topk}.txt')

    if not path.isdir(matches_path):
        os.makedirs(matches_path)
    if not path.isdir(matches_gv_path):
        os.makedirs(matches_gv_path)

    # build proxy kapture in output folder
    proxy_kapture_path = path.join(colmap_map_path, 'kapture_inputs/proxy_mapping')
    create_kapture_proxy(proxy_kapture_path,
                         kapture_path,
                         keypoints_path,
                         descriptors_path,
                         global_features_path,
                         matches_path,
                         force_overwrite_existing)

    # kapture_compute_image_pairs.py
    if 'compute_image_pairs' not in skip_list:
        local_image_pairs_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_compute_image_pairs.py')
        if os.path.isfile(pairfile_path):
            safe_remove_file(pairfile_path, force_overwrite_existing)

        compute_image_pairs_args = ['-v', str(logger.level),
                                    '--mapping', proxy_kapture_path,
                                    '--query', proxy_kapture_path,
                                    '--topk', str(topk),
                                    '-o', pairfile_path]
        run_python_command(local_image_pairs_path, compute_image_pairs_args, python_binary)

    # kapture_compute_matches.py
    if 'compute_matches' not in skip_list:
        local_compute_matches_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_compute_matches.py')
        compute_matches_args = ['-v', str(logger.level),
                                '-i', proxy_kapture_path,
                                '--pairsfile-path', pairfile_path]
        run_python_command(local_compute_matches_path, compute_matches_args, python_binary)

    # build proxy gv kapture in output folder
    proxy_kapture_gv_path = path.join(colmap_map_path, 'kapture_inputs/proxy_mapping_gv')
    create_kapture_proxy(proxy_kapture_gv_path,
                         kapture_path,
                         keypoints_path,
                         descriptors_path,
                         global_features_path,
                         matches_gv_path,
                         force_overwrite_existing)

    # kapture_run_colmap_gv.py
    if 'geometric_verification' not in skip_list:
        local_run_colmap_gv_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_run_colmap_gv.py')
        run_colmap_gv_args = ['-v', str(logger.level),
                              '-i', proxy_kapture_path,
                              '-o', proxy_kapture_gv_path,
                              '--pairsfile-path', pairfile_path,
                              '-colmap', colmap_binary]
        if force_overwrite_existing:
            run_colmap_gv_args.append('-f')
        run_python_command(local_run_colmap_gv_path, run_colmap_gv_args, python_binary)

    # kapture_colmap_build_map.py
    if 'colmap_build_map' not in skip_list:
        local_build_map_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_colmap_build_map.py')
        build_map_args = ['-v', str(logger.level),
                          '-i', proxy_kapture_gv_path,
                          '-o', colmap_map_path,
                          '-colmap', colmap_binary,
                          '--pairs-file-path', pairfile_path]
        if force_overwrite_existing:
            build_map_args.append('-f')
        build_map_args += ['--Mapper.ba_refine_focal_length', '0',
                           '--Mapper.ba_refine_principal_point', '0',
                           '--Mapper.ba_refine_extra_params', '0']
        run_python_command(local_build_map_path, build_map_args, python_binary)


def mapping_pipeline_command_line():
    """
    Parse the command line arguments to build a colmap map using the given kapture data.
    """
    parser = argparse.ArgumentParser(description=('create a Colmap model (map) from data specified in kapture format.'
                                                  'kapture data must contain keypoints, descriptors and '
                                                  'global features'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='silently delete pairfile and colmap reconstruction if already exists.')
    parser.add_argument('-i', '--kapture-map', required=True,
                        help='input path to kapture mapping data root directory')
    parser.add_argument('-kpt', '--keypoints-path', required=True,
                        help='input path to the orphan keypoints folder')
    parser.add_argument('-desc', '--descriptors-path', required=True,
                        help='input path to the orphan descriptors folder')
    parser.add_argument('-gfeat', '--global-features-path', required=True,
                        help='input path to the orphan global features folder')
    parser.add_argument('-matches', '--matches-path', required=True,
                        help='input path to the orphan matches (no geometric verification) folder')
    parser.add_argument('-matches-gv', '--matches-gv-path', required=True,
                        help='input path to the orphan matches (with geometric verification) folder')
    parser.add_argument('--colmap-map', required=True,
                        help='output directory for the pairfile and colmap reconstruction.')
    parser.add_argument('-colmap', '--colmap_binary', required=False,
                        default="colmap",
                        help='full path to colmap binary '
                             '(default is "colmap", i.e. assume the binary'
                             ' is in the user PATH).')
    parser_python_bin = parser.add_mutually_exclusive_group()
    parser_python_bin.add_argument('-python', '--python_binary', required=False,
                                   default=None,
                                   help='full path to python binary '
                                   '(default is "None", i.e. assume the os'
                                   ' can infer the python binary from the files itself, shebang or extension).')
    parser_python_bin.add_argument('--auto-python-binary', action='store_true', default=False,
                                   help='use sys.executable as python binary.')
    parser.add_argument('--topk',
                        default=20,
                        type=int,
                        help='the max number of top retained images when computing image pairs from global features')
    parser.add_argument('-s', '--skip', choices=['compute_image_pairs',
                                                 'compute_matches',
                                                 'geometric_verification',
                                                 'colmap_build_map'],
                        nargs='+', default=[],
                        help='steps to skip')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('mapping.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    if can_use_symlinks():
        python_binary = args.python_binary
        if args.auto_python_binary:
            python_binary = sys.executable
            logger.debug(f'python_binary set to {python_binary}')
        mapping_pipeline(args.kapture_map,
                         args.keypoints_path,
                         args.descriptors_path,
                         args.global_features_path,
                         args.matches_path,
                         args.matches_gv_path,
                         args.colmap_map,
                         args.colmap_binary,
                         python_binary,
                         args.topk,
                         args.skip,
                         args.force)
    else:
        raise EnvironmentError('Please restart this command as admin, it is required for os.symlink'
                               'see https://docs.python.org/3.6/library/os.html#os.symlink')
        # need to find a way to redirect output, else it closes on error...
        # logger.critical('Request UAC for symlink rights...')
        # ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)


if __name__ == '__main__':
    mapping_pipeline_command_line()
