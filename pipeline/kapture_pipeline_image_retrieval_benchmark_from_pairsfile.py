#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script runs the image retrieval benchmark
"""

import argparse
import logging
import os
import os.path as path
import sys
from typing import List, Optional

import pipeline_import_paths  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.symlink import can_use_symlinks, create_kapture_proxy_single_features
from kapture_localization.utils.subprocess import run_python_command
from kapture_localization.colmap.colmap_command import CONFIGS
from kapture_localization.utils.BenchmarkFormatStyle import BenchmarkFormatStyle, get_benchmark_format_command

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging

logger = logging.getLogger('image_retrieval_benchmark_from_pairsfile_pipeline')


def image_retrieval_benchmark_from_pairsfile(kapture_map_path: str,
                                             kapture_query_path: str,
                                             merge_path: Optional[str],
                                             pairsfile_path: str,
                                             keypoints_path: str,
                                             descriptors_path: str,
                                             matches_path: str,
                                             matches_gv_path: str,
                                             keypoints_type: Optional[str],
                                             descriptors_type: Optional[str],
                                             colmap_map_path: str,
                                             localization_output_path: str,
                                             colmap_binary: str,
                                             python_binary: Optional[str],
                                             config: int,
                                             benchmark_format_style: BenchmarkFormatStyle,
                                             skip_list: List[str],
                                             force_overwrite_existing: bool) -> None:
    """
    Image retrieval benchmark from pairsfile instead of global features (BDI not included)


    :param kapture_map_path: path to the kapture map directory
    :type kapture_map_path: str
    :param kapture_query_path: path to the kapture query directory
    :type kapture_query_path: str
    :param pairsfile_path: path to the file which contains the list of pairs
    :param merge_path: path to the kapture map+query directory
    :type merge_path: Optional[str]
    :param keypoints_path: input path to the orphan keypoints folder
    :type keypoints_path: str
    :param descriptors_path: input path to the orphan descriptors folder
    :type descriptors_path: str
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
    :param config: index of the config parameters to use for image registrator
    :type config: int
    :param benchmark_format_style: LTVL2020/RIO10 format output style
    :param skip_list: list of steps to ignore
    :type skip_list: List[str]
    :param force_overwrite_existing: silently overwrite files if already exists
    :type force_overwrite_existing: bool
    """
    os.makedirs(localization_output_path, exist_ok=True)
    map_plus_query_path = path.join(localization_output_path,
                                    'kapture_inputs/map_plus_query') if merge_path is None else merge_path
    eval_path = path.join(localization_output_path, f'eval')

    # global sfm results
    global_sfm_path = path.join(localization_output_path, f'global_sfm')
    global_sfm_colmap_localize_path = path.join(global_sfm_path, f'colmap_localized')
    os.makedirs(global_sfm_colmap_localize_path, exist_ok=True)
    global_sfm_kapture_localize_import_path = path.join(global_sfm_path, f'kapture_localized')
    global_sfm_kapture_localize_recover_path = path.join(global_sfm_path, f'kapture_localized_recover')
    global_sfm_LTVL2020_output_path = path.join(localization_output_path, 'global_sfm_LTVL2020_style_result.txt')

    # local sfm results
    local_sfm_path = path.join(localization_output_path, f'local_sfm')
    os.makedirs(local_sfm_path, exist_ok=True)
    local_sfm_localize_path = path.join(local_sfm_path, f'localized')
    local_sfm_LTVL2020_output_path = path.join(localization_output_path, 'local_sfm_LTVL2020_style_result.txt')

    # pose approximation results
    pose_approx_path = path.join(localization_output_path, f'pose_approx')
    pose_approx_EWB_path = path.join(pose_approx_path, f'EWB')
    pose_approx_CSI_path = path.join(pose_approx_path, f'CSI')

    pose_approx_EWB_LTVL2020_output_path = path.join(localization_output_path, 'EWB_LTVL2020_style_result.txt')
    pose_approx_CSI_LTVL2020_output_path = path.join(localization_output_path, 'CSI_LTVL2020_style_result.txt')

    if not path.isdir(matches_path):
        os.makedirs(matches_path)
    if not path.isdir(matches_gv_path):
        os.makedirs(matches_gv_path)

    # build proxy kapture map in output folder
    proxy_kapture_map_path = path.join(localization_output_path, 'kapture_inputs/proxy_mapping')
    create_kapture_proxy_single_features(proxy_kapture_map_path,
                                         kapture_map_path,
                                         keypoints_path,
                                         descriptors_path,
                                         None,
                                         matches_path,
                                         keypoints_type,
                                         descriptors_type,
                                         None,
                                         force_overwrite_existing)

    # build proxy kapture query in output folder
    proxy_kapture_query_path = path.join(localization_output_path, 'kapture_inputs/proxy_query')
    create_kapture_proxy_single_features(proxy_kapture_query_path,
                                         kapture_query_path,
                                         keypoints_path,
                                         descriptors_path,
                                         None,
                                         matches_path,
                                         keypoints_type,
                                         descriptors_type,
                                         None,
                                         force_overwrite_existing)

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
    create_kapture_proxy_single_features(proxy_kapture_map_plus_query_path,
                                         map_plus_query_path,
                                         keypoints_path,
                                         descriptors_path,
                                         None,
                                         matches_path,
                                         keypoints_type,
                                         descriptors_type,
                                         None,
                                         force_overwrite_existing)

    # kapture_compute_matches.py
    if 'compute_matches' not in skip_list:
        local_compute_matches_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_compute_matches.py')
        compute_matches_args = ['-v', str(logger.level),
                                '-i', proxy_kapture_map_plus_query_path,
                                '--pairsfile-path', pairsfile_path]
        run_python_command(local_compute_matches_path, compute_matches_args, python_binary)

    # build proxy gv kapture in output folder
    proxy_kapture_map_plus_query_gv_path = path.join(localization_output_path, 'kapture_inputs/proxy_map_plus_query_gv')
    create_kapture_proxy_single_features(proxy_kapture_map_plus_query_gv_path,
                                         map_plus_query_path,
                                         keypoints_path,
                                         descriptors_path,
                                         None,
                                         matches_gv_path,
                                         keypoints_type,
                                         descriptors_type,
                                         None,
                                         force_overwrite_existing)

    # kapture_run_colmap_gv.py
    if 'geometric_verification' not in skip_list:
        local_run_colmap_gv_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_run_colmap_gv.py')
        run_colmap_gv_args = ['-v', str(logger.level),
                              '-i', proxy_kapture_map_plus_query_path,
                              '-o', proxy_kapture_map_plus_query_gv_path,
                              '--pairsfile-path', pairsfile_path,
                              '-colmap', colmap_binary]
        if force_overwrite_existing:
            run_colmap_gv_args.append('-f')
        run_python_command(local_run_colmap_gv_path, run_colmap_gv_args, python_binary)

    # -------- GLOBAL MAP LOCALIZATION --------
    if 'global_sfm' not in skip_list:
        # kapture_colmap_localize.py
        local_localize_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_colmap_localize.py')
        localize_args = ['-v', str(logger.level),
                         '-i', proxy_kapture_map_plus_query_gv_path,
                         '-o', global_sfm_colmap_localize_path,
                         '-colmap', colmap_binary,
                         '--pairs-file-path', pairsfile_path,
                         '-db', path.join(colmap_map_path, 'colmap.db'),
                         '-txt', path.join(colmap_map_path, 'reconstruction')]
        if force_overwrite_existing:
            localize_args.append('-f')
        localize_args += CONFIGS[config]
        run_python_command(local_localize_path, localize_args, python_binary)

        # kapture_import_colmap.py
        local_import_colmap_path = path.join(pipeline_import_paths.HERE_PATH,
                                             '../../kapture/tools/kapture_import_colmap.py')
        import_colmap_args = ['-v', str(logger.level),
                              '-db', path.join(global_sfm_colmap_localize_path, 'colmap.db'),
                              '-txt', path.join(global_sfm_colmap_localize_path, 'reconstruction'),
                              '-o', global_sfm_kapture_localize_import_path,
                              '--skip_reconstruction']
        if force_overwrite_existing:
            import_colmap_args.append('-f')
        run_python_command(local_import_colmap_path, import_colmap_args, python_binary)

        local_recover_path = path.join(pipeline_import_paths.HERE_PATH,
                                       '../tools/kapture_recover_timestamps_and_ids.py')
        recover_args = ['-v', str(logger.level),
                        '-i', global_sfm_kapture_localize_import_path,
                        '--ref', proxy_kapture_query_path,
                        '-o', global_sfm_kapture_localize_recover_path,
                        '--image_transfer', 'skip']
        if force_overwrite_existing:
            recover_args.append('-f')
        run_python_command(local_recover_path, recover_args, python_binary)

        # kapture_export_LTVL2020.py
        if 'export_LTVL2020' not in skip_list:
            export_LTVL2020_script_name, export_LTVL2020_args = get_benchmark_format_command(
                benchmark_format_style,
                global_sfm_kapture_localize_recover_path,
                global_sfm_LTVL2020_output_path,
                force_overwrite_existing,
                logger
            )
            local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                                   f'../../kapture/tools/{export_LTVL2020_script_name}')
            run_python_command(local_export_LTVL2020_path, export_LTVL2020_args, python_binary)

    # -------- LOCAL SFM LOCALIZATION --------
    if 'local_sfm' not in skip_list:
        # kapture_colmap_localize_localsfm
        local_colmap_localize_localsfm_path = path.join(pipeline_import_paths.HERE_PATH,
                                                        '../tools/kapture_colmap_localize_localsfm.py')
        colmap_localize_localsfm_args = ['-v', str(logger.level),
                                         '--map_plus_query', proxy_kapture_map_plus_query_path,
                                         '--map_plus_query_gv', proxy_kapture_map_plus_query_gv_path,
                                         '--query', proxy_kapture_query_path,
                                         '-o', local_sfm_path,
                                         '-colmap', colmap_binary,
                                         '--pairsfile-path', pairsfile_path]
        if force_overwrite_existing:
            colmap_localize_localsfm_args.append('-f')
        run_python_command(local_colmap_localize_localsfm_path, colmap_localize_localsfm_args, python_binary)

        # kapture_export_LTVL2020.py
        if 'export_LTVL2020' not in skip_list:
            export_LTVL2020_script_name, export_LTVL2020_args = get_benchmark_format_command(
                benchmark_format_style,
                local_sfm_localize_path,
                local_sfm_LTVL2020_output_path,
                force_overwrite_existing,
                logger
            )
            local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                                   f'../../kapture/tools/{export_LTVL2020_script_name}')
            run_python_command(local_export_LTVL2020_path, export_LTVL2020_args, python_binary)

    # -------- POSE APPROXIMATION LOCALIZATION --------
    if 'pose_approximation' not in skip_list:
        # kapture_pose_approximation.py
        local_pose_approximation_path = path.join(pipeline_import_paths.HERE_PATH,
                                                  '../tools/kapture_pose_approximation_from_pairsfile.py')
        pose_approximation_args = ['-v', str(logger.level),
                                   '--mapping', proxy_kapture_map_path,
                                   '--query', proxy_kapture_query_path,
                                   '--pairsfile-path', pairsfile_path]
        if force_overwrite_existing:
            pose_approximation_args.append('-f')

        # EWB
        EWB_pose_approximation_args = pose_approximation_args + ['-o', pose_approx_EWB_path,
                                                                 'equal_weighted_barycenter']
        run_python_command(local_pose_approximation_path, EWB_pose_approximation_args, python_binary)

        # CSI
        CSI_pose_approximation_args = pose_approximation_args + ['-o', pose_approx_CSI_path,
                                                                 'cosine_similarity']
        run_python_command(local_pose_approximation_path, CSI_pose_approximation_args, python_binary)

        # kapture_export_LTVL2020.py
        if 'export_LTVL2020' not in skip_list:
            EWB_export_LTVL2020_script_name, EWB_export_LTVL2020_args = get_benchmark_format_command(
                benchmark_format_style,
                pose_approx_EWB_path,
                pose_approx_EWB_LTVL2020_output_path,
                force_overwrite_existing,
                logger
            )
            EWB_local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                                       f'../../kapture/tools/{EWB_export_LTVL2020_script_name}')
            run_python_command(EWB_local_export_LTVL2020_path, EWB_export_LTVL2020_args, python_binary)

            CSI_export_LTVL2020_script_name, CSI_export_LTVL2020_args = get_benchmark_format_command(
                benchmark_format_style,
                pose_approx_CSI_path,
                pose_approx_CSI_LTVL2020_output_path,
                force_overwrite_existing,
                logger
            )
            CSI_local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                                       f'../../kapture/tools/{CSI_export_LTVL2020_script_name}')
            run_python_command(CSI_local_export_LTVL2020_path, CSI_export_LTVL2020_args, python_binary)

    # -------- EVALUATE ALL AT ONCE --------
    # kapture_evaluate.py
    if 'evaluate' not in skip_list and path.isfile(path.join(kapture_query_path, 'sensors/trajectories.txt')):
        local_evaluate_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_evaluate.py')
        input_list = []
        label_list = []
        if os.path.isdir(global_sfm_kapture_localize_recover_path):
            input_list.append(global_sfm_kapture_localize_recover_path)
            label_list.append(f'global_sfm_config_{config}')
        if os.path.isdir(local_sfm_localize_path):
            input_list.append(local_sfm_localize_path)
            label_list.append('local_sfm')
        if os.path.isdir(pose_approx_EWB_path):
            input_list.append(pose_approx_EWB_path)
            label_list.append('EWB')
        if os.path.isdir(pose_approx_CSI_path):
            input_list.append(pose_approx_CSI_path)
            label_list.append('CSI')
        evaluate_args = ['-v', str(logger.level),
                         '-i'] + input_list + ['--labels'] + label_list + ['-gt', kapture_query_path,
                                                                           '-o', eval_path]
        if force_overwrite_existing:
            evaluate_args.append('-f')
        run_python_command(local_evaluate_path, evaluate_args, python_binary)


def image_retrieval_benchmark_from_pairsfile_get_parser():
    """
    get the argparse object for the kapture_pipeline_image_retrieval_benchmark_from_pairsfile.py command
    """
    parser = argparse.ArgumentParser(description='run the image retrieval benchmark on kapture data')
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
    parser.add_argument('--pairsfile-path', required=True, type=str,
                        help='text file which contains the image pairs and their score')
    parser.add_argument('-kpt', '--keypoints-path', required=True,
                        help='input path to the orphan keypoints folder')
    parser.add_argument('-desc', '--descriptors-path', required=True,
                        help='input path to the orphan descriptors folder')
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
    parser.add_argument('--config', default=1, type=int,
                        choices=list(range(len(CONFIGS))), help='what config to use for global sfm image registrator')
    parser.add_argument('--benchmark-style',
                        default=BenchmarkFormatStyle.Default,
                        type=BenchmarkFormatStyle,
                        choices=list(BenchmarkFormatStyle),
                        help=('select which output format to use for the export_LTVL2020 part.'
                              ' Default is the https://www.visuallocalization.net default.'
                              ' RobotCar_Seasons, Gangnam_Station, Hyundai_Department_Store,'
                              ' ETH_Microsoft are also part of'
                              ' https://www.visuallocalization.net but require a different format.'
                              ' RIO10 is for http://vmnavab26.in.tum.de/RIO10/'))
    parser.add_argument('-s', '--skip', choices=['compute_matches',
                                                 'geometric_verification',
                                                 'global_sfm',
                                                 'local_sfm',
                                                 'pose_approximation',
                                                 'evaluate',
                                                 'export_LTVL2020'],
                        nargs='+', default=[],
                        help='steps to skip')
    parser.add_argument('--keypoints-type', default=None, help='kapture keypoints type.')
    parser.add_argument('--descriptors-type', default=None, help='kapture descriptors type.')
    return parser


def image_retrieval_benchmark_from_pairsfile_command_line():
    """
    Parse the command line arguments to run the image retrieval benchmark on kapture data.
    """
    parser = image_retrieval_benchmark_from_pairsfile_get_parser()
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('image_retrieval_benchmark_from_pairsfile.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))
    if can_use_symlinks():
        python_binary = args.python_binary
        if args.auto_python_binary:
            python_binary = sys.executable
            logger.debug(f'python_binary set to {python_binary}')
        image_retrieval_benchmark_from_pairsfile(args.kapture_map,
                                                 args.query,
                                                 args.merge_path,
                                                 args.pairsfile_path,
                                                 args.keypoints_path,
                                                 args.descriptors_path,
                                                 args.matches_path,
                                                 args.matches_gv_path,
                                                 args.keypoints_type,
                                                 args.descriptors_type,
                                                 args.colmap_map,
                                                 args.output,
                                                 args.colmap_binary,
                                                 python_binary,
                                                 args.config,
                                                 args.benchmark_style,
                                                 args.skip,
                                                 args.force)
    else:
        raise EnvironmentError('Please restart this command as admin, it is required for os.symlink'
                               'see https://docs.python.org/3.6/library/os.html#os.symlink')
        # need to find a way to redirect output, else it closes on error...
        # logger.critical('Request UAC for symlink rights...')
        # ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)


if __name__ == '__main__':
    image_retrieval_benchmark_from_pairsfile_command_line()
