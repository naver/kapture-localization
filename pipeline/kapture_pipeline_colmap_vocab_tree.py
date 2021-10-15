#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script builds a COLMAP model (map) from kapture format (images, cameras, trajectories) with colmap sift+vocab tree
"""

import argparse
import logging
import os
import os.path as path
import sys
from typing import List, Optional

import pipeline_import_paths  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.subprocess import run_python_command
from kapture_localization.colmap.colmap_command import CONFIGS
from kapture_localization.utils.BenchmarkFormatStyle import BenchmarkFormatStyle, get_benchmark_format_command

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging

logger = logging.getLogger('colmap_vocab_tree_pipeline')


def colmap_vocab_tree_pipeline(kapture_map_path: str,
                               kapture_query_path: Optional[str],
                               colmap_map_path: Optional[str],
                               localization_output_path: str,
                               colmap_binary: str,
                               python_binary: Optional[str],
                               vocab_tree_path: str,
                               mapping_config: int,
                               localize_config: int,
                               benchmark_format_style: BenchmarkFormatStyle,
                               bins_as_str: List[str],
                               skip_list: List[str],
                               force_overwrite_existing: bool) -> None:
    """
    Build a colmap model using sift features and vocab tree matching features with the kapture data.

    :param kapture_map_path: path to the kapture map directory
    :type kapture_map_path: str
    :param kapture_query_path: path to the kapture query directory
    :type kapture_query_path: Optional[str]
    :param colmap_map_path: input path to the colmap reconstruction folder
    :type colmap_map_path: Optional[str]
    :param localization_output_path: output path to the localization results
    :type localization_output_path: str
    :param colmap_binary: path to the colmap executable
    :type colmap_binary: str
    :param python_binary: path to the python executable
    :type python_binary: Optional[str]
    :param vocab_tree_path: full path to Vocabulary Tree file used for matching
    :type vocab_tree_path: str
    :param mapping_config: index of the config parameters to use for point triangulator
    :type mapping_config: int
    :param localize_config: index of the config parameters to use for image registrator
    :type localize_config: int
    :param benchmark_format_style: LTVL2020/RIO10 format output style
    :param bins_as_str: list of bin names
    :type bins_as_str: List[str]
    :param skip_list: list of steps to ignore
    :type skip_list: List[str]
    :param force_overwrite_existing: silently overwrite files if already exists
    :type force_overwrite_existing: bool
    """
    if colmap_map_path is None:
        colmap_map_path = path.join(localization_output_path, 'colmap_map')
        os.makedirs(colmap_map_path, exist_ok=True)
    elif 'colmap_build_sift_map' not in skip_list:
        logger.info('--colmap-map is not None, reusing existing map, skipping colmap_build_sift_map')
        skip_list.append('colmap_build_sift_map')

    # kapture_colmap_build_sift_map.py
    if 'colmap_build_sift_map' not in skip_list:
        local_build_sift_map_path = path.join(pipeline_import_paths.HERE_PATH,
                                              '../tools/kapture_colmap_build_sift_map.py')

        build_sift_map_args = ['-v', str(logger.level),
                               '-i', kapture_map_path,
                               '-o', colmap_map_path,
                               '-voc', vocab_tree_path,
                               '-colmap', colmap_binary]
        if force_overwrite_existing:
            build_sift_map_args.append('-f')
        build_sift_map_args += CONFIGS[mapping_config]
        run_python_command(local_build_sift_map_path, build_sift_map_args, python_binary)

    if kapture_query_path is None:
        return

    colmap_localize_path = path.join(localization_output_path, f'colmap_localized')
    os.makedirs(colmap_localize_path, exist_ok=True)
    kapture_localize_import_path = path.join(localization_output_path, f'kapture_localized')
    kapture_localize_recover_path = path.join(localization_output_path, f'kapture_localized_recover')
    eval_path = path.join(localization_output_path, f'eval')
    LTVL2020_output_path = path.join(localization_output_path, 'LTVL2020_style_result.txt')

    # kapture_colmap_localize_sift.py
    if 'colmap_localize_sift' not in skip_list:
        local_localize_sift_path = path.join(pipeline_import_paths.HERE_PATH,
                                             '../tools/kapture_colmap_localize_sift.py')
        localize_sift_args = ['-v', str(logger.level),
                              '-i', kapture_query_path,
                              '-db', path.join(colmap_map_path, 'colmap.db'),
                              '-txt', path.join(colmap_map_path, 'reconstruction'),
                              '-o', colmap_localize_path,
                              '-voc', vocab_tree_path,
                              '-colmap', colmap_binary]
        if force_overwrite_existing:
            localize_sift_args.append('-f')
        localize_sift_args += CONFIGS[localize_config]
        run_python_command(local_localize_sift_path, localize_sift_args, python_binary)

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

        local_recover_path = path.join(pipeline_import_paths.HERE_PATH,
                                       '../tools/kapture_recover_timestamps_and_ids.py')
        recover_args = ['-v', str(logger.level),
                        '-i', kapture_localize_import_path,
                        '--ref', kapture_query_path,
                        '-o', kapture_localize_recover_path,
                        '--image_transfer', 'skip']
        if force_overwrite_existing:
            recover_args.append('-f')
        run_python_command(local_recover_path, recover_args, python_binary)

    # kapture_evaluate.py
    if 'evaluate' not in skip_list and path.isfile(path.join(kapture_query_path, 'sensors/trajectories.txt')):
        local_evaluate_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_evaluate.py')
        evaluate_args = ['-v', str(logger.level),
                         '-i', kapture_localize_recover_path,
                         '--labels', f'sift_colmap_vocab_tree_config_{localize_config}',
                         '-gt', kapture_query_path,
                         '-o', eval_path]
        evaluate_args += ['--bins'] + bins_as_str
        if force_overwrite_existing:
            evaluate_args.append('-f')
        run_python_command(local_evaluate_path, evaluate_args, python_binary)

    # kapture_export_LTVL2020.py
    if 'export_LTVL2020' not in skip_list:
        export_LTVL2020_script_name, export_LTVL2020_args = get_benchmark_format_command(
            benchmark_format_style,
            kapture_localize_recover_path,
            LTVL2020_output_path,
            force_overwrite_existing,
            logger
        )
        local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                               f'../../kapture/tools/{export_LTVL2020_script_name}')
        run_python_command(local_export_LTVL2020_path, export_LTVL2020_args, python_binary)


def colmap_vocab_tree_pipeline_get_parser():
    """
    get the argparse object for the kapture_pipeline_colmap_vocab_tree.py command
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
    parser.add_argument('--query', default=None,
                        help='input path to kapture mapping data root directory')
    parser.add_argument('--colmap-map', default=None,
                        help='path to the input colmap map directory (will be computed when left to None)')
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
    parser.add_argument('-voc', '--vocab_tree_path', required=True,
                        help='full path to Vocabulary Tree file'
                             ' used for matching.')
    parser.add_argument('--mapping-config', default=1, type=int,
                        choices=[0, 1], help='what config to use for point triangulator')
    parser.add_argument('--localize-config', default=1, type=int,
                        choices=list(range(len(CONFIGS))), help='what config to use for image registrator')
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
    parser.add_argument('--bins', nargs='+', default=["0.25 2", "0.5 5", "5 10"],
                        help='the desired positions/rotations thresholds for bins'
                        'format is string : position_threshold_in_m space rotation_threshold_in_degree')
    parser.add_argument('-s', '--skip', choices=['colmap_build_sift_map'
                                                 'colmap_localize_sift.py',
                                                 'import_colmap',
                                                 'evaluate',
                                                 'export_LTVL2020'],
                        nargs='+', default=[],
                        help='steps to skip')
    return parser


def colmap_vocab_tree_pipeline_command_line():
    """
    Parse the command line arguments to build a map and localize images using colmap on the given kapture data.
    """
    parser = colmap_vocab_tree_pipeline_get_parser()
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('localize.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))
    python_binary = args.python_binary
    if args.auto_python_binary:
        python_binary = sys.executable
        logger.debug(f'python_binary set to {python_binary}')
    colmap_vocab_tree_pipeline(args.kapture_map,
                               args.query,
                               args.colmap_map,
                               args.output,
                               args.colmap_binary,
                               python_binary,
                               args.vocab_tree_path,
                               args.mapping_config,
                               args.localize_config,
                               args.benchmark_style,
                               args.bins,
                               args.skip,
                               args.force)


if __name__ == '__main__':
    colmap_vocab_tree_pipeline_command_line()
