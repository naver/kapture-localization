#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script localize images on an existing COLMAP model (map)
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
from kapture_localization.colmap.colmap_command import CONFIGS

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging
from kapture.utils.paths import safe_remove_file

logger = logging.getLogger('localization_pipeline')


def localize_pipeline(kapture_map_path: str,
                      kapture_query_path: str,
                      merge_path: Optional[str],
                      keypoints_path: str,
                      descriptors_path: str,
                      global_features_path: str,
                      matches_path: str,
                      matches_gv_path: str,
                      colmap_map_path: str,
                      localization_output_path: str,
                      colmap_binary: str,
                      python_binary: Optional[str],
                      topk: int,
                      config: int,
                      prepend_cam: bool,
                      bins_as_str: List[str],
                      skip_list: List[str],
                      force_overwrite_existing: bool) -> None:
    """
    Localize on colmap map

    :param kapture_map_path: path to the kapture map directory
    :type kapture_map_path: str
    :param kapture_query_path: path to the kapture query directory
    :type kapture_query_path: str
    :param merge_path: path to the kapture map+query directory
    :type merge_path: Optional[str]
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
    :param colmap_map_path: input path to the colmap reconstruction folder
    :type colmap_map_path: str
    :param localization_output_path: output path to the localization results
    :type localization_output_path: str
    :param colmap_binary: path to the colmap executable
    :type colmap_binary: str
    :param python_binary: path to the python executable
    :type python_binary: Optional[str]
    :param topk: the max number of top retained images when computing image pairs from global features
    :type topk: int
    :param config: index of the config parameters to use for image registrator
    :type config: int
    :param prepend_cam: prepend camera names to filename in LTVL2020 formatted output
    :type prepend_cam: bool
    :param bins_as_str: list of bin names
    :type bins_as_str: List[str]
    :param skip_list: list of steps to ignore
    :type skip_list: List[str]
    :param force_overwrite_existing: silently overwrite files if already exists
    :type force_overwrite_existing: bool
    """
    os.makedirs(localization_output_path, exist_ok=True)
    pairfile_path = path.join(localization_output_path, f'pairs_localization_{topk}.txt')
    map_plus_query_path = path.join(localization_output_path,
                                    'kapture_inputs/map_plus_query') if merge_path is None else merge_path
    colmap_localize_path = path.join(localization_output_path, f'colmap_localized')
    os.makedirs(colmap_localize_path, exist_ok=True)
    kapture_localize_import_path = path.join(localization_output_path, f'kapture_localized')
    eval_path = path.join(localization_output_path, f'eval')
    LTVL2020_output_path = path.join(localization_output_path, 'LTVL2020_style_result.txt')

    if not path.isdir(matches_path):
        os.makedirs(matches_path)
    if not path.isdir(matches_gv_path):
        os.makedirs(matches_gv_path)

    # build proxy kapture map in output folder
    proxy_kapture_map_path = path.join(localization_output_path, 'kapture_inputs/proxy_mapping')
    create_kapture_proxy(proxy_kapture_map_path,
                         kapture_map_path,
                         keypoints_path,
                         descriptors_path,
                         global_features_path,
                         matches_path,
                         force_overwrite_existing)

    # build proxy kapture query in output folder
    proxy_kapture_query_path = path.join(localization_output_path, 'kapture_inputs/proxy_query')
    create_kapture_proxy(proxy_kapture_query_path,
                         kapture_query_path,
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
                                    '--mapping', proxy_kapture_map_path,
                                    '--query', proxy_kapture_query_path,
                                    '--topk', str(topk),
                                    '-o', pairfile_path]
        run_python_command(local_image_pairs_path, compute_image_pairs_args, python_binary)

    # kapture_merge.py
    if merge_path is None:
        local_merge_path = path.join(pipeline_import_paths.HERE_PATH, '../../kapture/tools/kapture_merge.py')
        merge_args = ['-v', str(logger.level),
                      '-i', proxy_kapture_map_path, proxy_kapture_query_path,
                      '-o', map_plus_query_path,
                      '-s', 'keypoints', 'descriptors', 'global_features', 'matches',
                      '--image_transfer', 'link_absolute']
        if force_overwrite_existing:
            merge_args.append('-f')
        run_python_command(local_merge_path, merge_args, python_binary)

    # build proxy kapture map+query in output folder
    proxy_kapture_map_plus_query_path = path.join(localization_output_path, 'kapture_inputs/proxy_map_plus_query')
    create_kapture_proxy(proxy_kapture_map_plus_query_path,
                         map_plus_query_path,
                         keypoints_path,
                         descriptors_path,
                         global_features_path,
                         matches_path,
                         force_overwrite_existing)

    # kapture_compute_matches.py
    if 'compute_matches' not in skip_list:
        local_compute_matches_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_compute_matches.py')
        compute_matches_args = ['-v', str(logger.level),
                                '-i', proxy_kapture_map_plus_query_path,
                                '--pairsfile-path', pairfile_path]
        run_python_command(local_compute_matches_path, compute_matches_args, python_binary)

    # build proxy gv kapture in output folder
    proxy_kapture_map_plus_query_gv_path = path.join(localization_output_path, 'kapture_inputs/proxy_map_plus_query_gv')
    create_kapture_proxy(proxy_kapture_map_plus_query_gv_path,
                         map_plus_query_path,
                         keypoints_path,
                         descriptors_path,
                         global_features_path,
                         matches_gv_path,
                         force_overwrite_existing)

    # kapture_run_colmap_gv.py
    if 'geometric_verification' not in skip_list:
        local_run_colmap_gv_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_run_colmap_gv.py')
        run_colmap_gv_args = ['-v', str(logger.level),
                              '-i', proxy_kapture_map_plus_query_path,
                              '-o', proxy_kapture_map_plus_query_gv_path,
                              '--pairsfile-path', pairfile_path,
                              '-colmap', colmap_binary]
        if force_overwrite_existing:
            run_colmap_gv_args.append('-f')
        run_python_command(local_run_colmap_gv_path, run_colmap_gv_args, python_binary)

    # kapture_colmap_localize.py
    if 'colmap_localize' not in skip_list:
        local_localize_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_colmap_localize.py')
        localize_args = ['-v', str(logger.level),
                         '-i', proxy_kapture_map_plus_query_gv_path,
                         '-o', colmap_localize_path,
                         '-colmap', colmap_binary,
                         '--pairs-file-path', pairfile_path,
                         '-db', path.join(colmap_map_path, 'colmap.db'),
                         '-txt', path.join(colmap_map_path, 'reconstruction')]
        if force_overwrite_existing:
            localize_args.append('-f')
        localize_args += CONFIGS[config]
        run_python_command(local_localize_path, localize_args, python_binary)

    # kapture_import_colmap.py
    if 'import_colmap' not in skip_list:
        local_import_colmap_path = path.join(pipeline_import_paths.HERE_PATH,
                                             '../../kapture/tools/kapture_import_colmap.py')
        import_colmap_args = ['-v', str(logger.level),
                              '-db', path.join(colmap_localize_path, 'colmap.db'),
                              '-txt', path.join(colmap_localize_path, 'reconstruction'),
                              '-o', kapture_localize_import_path,
                              '--skip_reconstruction']
        if force_overwrite_existing:
            import_colmap_args.append('-f')
        run_python_command(local_import_colmap_path, import_colmap_args, python_binary)

    # kapture_evaluate.py
    if 'evaluate' not in skip_list and path.isfile(path.join(kapture_query_path, 'sensors/trajectories.txt')):
        local_evaluate_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_evaluate.py')
        evaluate_args = ['-v', str(logger.level),
                         '-i', kapture_localize_import_path,
                         '--labels', f'colmap_config_{config}',
                         '-gt', kapture_query_path,
                         '-o', eval_path]
        evaluate_args += ['--bins'] + bins_as_str
        if force_overwrite_existing:
            evaluate_args.append('-f')
        run_python_command(local_evaluate_path, evaluate_args, python_binary)

    # kapture_export_LTVL2020.py
    if 'export_LTVL2020' not in skip_list:
        local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                               '../../kapture/tools/kapture_export_LTVL2020.py')
        export_LTVL2020_args = ['-v', str(logger.level),
                                '-i', kapture_localize_import_path,
                                '-o', LTVL2020_output_path]
        if prepend_cam:
            export_LTVL2020_args.append('-p')
        if force_overwrite_existing:
            export_LTVL2020_args.append('-f')
        run_python_command(local_export_LTVL2020_path, export_LTVL2020_args, python_binary)


def localize_pipeline_command_line():
    """
    Parse the command line arguments to localize images on a colmap map using the given kapture data.
    """
    parser = argparse.ArgumentParser(description='localize images given in kapture format on a colmap map')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='silently delete pairfile and localization results if already exists.')
    parser.add_argument('-i', '--kapture-map', required=True,
                        help='path to the kapture map directory')
    parser.add_argument('--query', required=True,
                        help='input path to kapture mapping data root directory')
    parser.add_argument('--merge-path', required=False, default=None,
                        help=('optional, path to the kapture map+query directory. '
                              'using this will skip the call to kapture_merge.py and save some time'))
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
                        help='path to the colmap map directory')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory.')
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
    parser.add_argument('--config', default=1, type=int,
                        choices=list(range(len(CONFIGS))), help='what config to use for image registrator')
    parser.add_argument('--prepend_cam', action='store_true', default=False,
                        help=('prepend camera names to filename in LTVL2020 formatted output. '
                              'Toggle this only for RobotCar_Seasons and RobotCar Seasons v2'))
    parser.add_argument('--bins', nargs='+', default=["0.25 2", "0.5 5", "5 10"],
                        help='the desired positions/rotations thresholds for bins'
                        'format is string : position_threshold_in_m space rotation_threshold_in_degree')
    parser.add_argument('-s', '--skip', choices=['compute_image_pairs'
                                                 'compute_matches',
                                                 'geometric_verification',
                                                 'colmap_localize',
                                                 'import_colmap',
                                                 'evaluate',
                                                 'export_LTVL2020'],
                        nargs='+', default=[],
                        help='steps to skip')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('localize.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))
    if can_use_symlinks():
        python_binary = args.python_binary
        if args.auto_python_binary:
            python_binary = sys.executable
            logger.debug(f'python_binary set to {python_binary}')
        localize_pipeline(args.kapture_map,
                          args.query,
                          args.merge_path,
                          args.keypoints_path,
                          args.descriptors_path,
                          args.global_features_path,
                          args.matches_path,
                          args.matches_gv_path,
                          args.colmap_map,
                          args.output,
                          args.colmap_binary,
                          python_binary,
                          args.topk,
                          args.config,
                          args.prepend_cam,
                          args.bins,
                          args.skip,
                          args.force)
    else:
        raise EnvironmentError('Please restart this command as admin, it is required for os.symlink'
                               'see https://docs.python.org/3.6/library/os.html#os.symlink')
        # need to find a way to redirect output, else it closes on error...
        # logger.critical('Request UAC for symlink rights...')
        # ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)


if __name__ == '__main__':
    localize_pipeline_command_line()
