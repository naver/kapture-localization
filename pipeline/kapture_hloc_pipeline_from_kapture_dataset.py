#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license


import argparse
import logging
import os
import os.path as path
import sys
from typing import List, Optional
from pathlib import Path
import numpy as np  # noqa: F401
import quaternion

try:
    from pprint import pformat  # noqa: F401
    import h5py  # noqa: F401
    import cv2  # noqa: F401
    from hloc import extract_features, match_features, pairs_from_covisibility  # noqa: F401
    from hloc import triangulation, localize_sfm, visualization  # noqa: F401
except Exception as e:
    raise ImportError(f' {e} hloc have additional requirements compared to kapture-localization, '
                      'please see https://github.com/cvg/Hierarchical-Localization/blob/master/requirements.txt '
                      'and add Hierarchical-Localization to your PYTHONPATH')


import pipeline_import_paths  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.subprocess import run_python_command
from kapture_localization.colmap.colmap_command import run_model_converter
from kapture_localization.utils.BenchmarkFormatStyle import BenchmarkFormatStyle, get_benchmark_format_command

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging
import kapture
from kapture.io.csv import table_from_file, kapture_from_dir, kapture_to_dir, get_csv_fullpath
from kapture.io.records import get_record_fullpath
from kapture.converter.colmap.export_colmap import export_colmap

logger = logging.getLogger('hloc_pipeline_from_kapture_dataset')


def convert_pairs_to_hloc_format(pairsfile_path_kapture: str, pairsfile_path_hloc: str):
    """
    convert kapture pairsfile to hloc pairsfile
    """
    with open(pairsfile_path_kapture, 'r') as fid:
        table = list(table_from_file(fid))
    os.makedirs(os.path.dirname(os.path.abspath(pairsfile_path_hloc)), exist_ok=True)
    with open(pairsfile_path_hloc, 'w') as fid:
        for query_name, map_name, _ in table:
            fid.write(f'{query_name} {map_name}\n')


def convert_kapture_to_hloc_image_list(kapture_path: str, output_path: str):
    """
    convert kapture records_camera to hloc image list
    """
    skip_heavy_useless = [kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_to_convert = kapture_from_dir(kapture_path, skip_list=skip_heavy_useless)
    output_content = []
    for _, sensor_id, filename in kapture.flatten(kapture_to_convert.records_camera, is_sorted=True):
        line = filename
        output_content.append(line)
    with open(output_path, 'w') as fid:
        fid.write('\n'.join(output_content))


def export_image_list(kapture_path: str, output_path: str):
    """
    export from kapture to image list with camera params
    """
    skip_heavy_useless = [kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_to_export = kapture_from_dir(kapture_path, skip_list=skip_heavy_useless)
    output_content = []
    for _, sensor_id, filename in kapture.flatten(kapture_to_export.records_camera, is_sorted=True):
        line = filename
        camera = kapture_to_export.sensors[sensor_id]
        line += ' ' + ' '.join(camera.sensor_params)
        output_content.append(line)
    with open(output_path, 'w') as fid:
        fid.write('\n'.join(output_content))


def convert_results_format(image_list_with_intrinsics_path: str, results_file_in: str, results_file_out: str):
    """
    convert hloc result file (with basename for images) to the same but with the full relative path
    """
    with open(image_list_with_intrinsics_path) as fid:
        images_list = fid.readlines()
        # remove end line char and empty lines
        images_list = [line.rstrip() for line in images_list if line != '\n']
        images_list = [line.split()[0] for line in images_list]
    with open(results_file_in) as fid:
        lines = fid.readlines()
        lines = [line.rstrip() for line in lines if line != '\n']
    with open(results_file_out, 'w') as fid:
        for i, line in enumerate(lines):
            line_array = line.split()
            line_array[0] = images_list[i]
            fid.write(' '.join(line_array) + '\n')


def convert_results_to_kapture(query_path: str, results: str, outpath: str):
    """
    convert file with name qw qx qy qz tx ty tz to kapture
    """
    skip_heavy_useless = [kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_query = kapture_from_dir(query_path, skip_list=skip_heavy_useless)
    inverse_records_camera = {image_name: (timestamp, sensor_id) for timestamp,
                              sensor_id, image_name in kapture.flatten(kapture_query.records_camera)}
    trajectories = kapture.Trajectories()
    with open(results) as fid:
        lines = fid.readlines()
        lines = [line.rstrip().split() for line in lines if line != '\n']
    for line in lines:
        image_name = line[0]
        rotation = quaternion.quaternion(float(line[1]), float(line[2]), float(line[3]), float(line[4]))
        translation = [float(line[5]), float(line[6]), float(line[7])]
        timestamp, sensor_id = inverse_records_camera[image_name]
        trajectories[timestamp, sensor_id] = kapture.PoseTransform(rotation, translation)
    kapture_query.trajectories = trajectories
    kapture_to_dir(outpath, kapture_query)


def hloc_pipeline_from_kapture_dataset(kapture_path_map: str,
                                       kapture_path_query: str,
                                       pairsfile_path_map: str,
                                       pairsfile_path_query: str,
                                       output_dir: str,
                                       feature_conf_str: str,
                                       matcher_conf_str: str,
                                       covisibility_clustering: bool,
                                       bins_as_str: List[str],
                                       benchmark_format_style: BenchmarkFormatStyle,
                                       colmap_binary: str,
                                       python_binary: Optional[str],
                                       skip_list: List[str]) -> None:
    """
    run hloc on kapture data
    """
    feature_conf = extract_features.confs[feature_conf_str]
    matcher_conf = match_features.confs[matcher_conf_str]
    images_map = get_record_fullpath(kapture_path_map)
    images_query = get_record_fullpath(kapture_path_query)

    os.makedirs(output_dir, exist_ok=True)
    if "convert_pairsfile_map" not in skip_list:
        map_pairs_hloc = path.join(output_dir, 'pairfiles/db_pairs', path.basename(pairsfile_path_map) + "_hloc.txt")
        convert_pairs_to_hloc_format(pairsfile_path_map, map_pairs_hloc)
        pairsfile_path_map = map_pairs_hloc
    if "convert_pairsfile_query" not in skip_list:
        query_pairs_hloc = path.join(output_dir, 'pairfiles/query', path.basename(pairsfile_path_query) + "_hloc.txt")
        convert_pairs_to_hloc_format(pairsfile_path_query, query_pairs_hloc)
        pairsfile_path_query = query_pairs_hloc

    feature_path = Path(output_dir, feature_conf['output']+'.h5')
    if "extract_features_map" not in skip_list:
        image_list_map_path = path.join(output_dir, 'image_list_map.txt')
        convert_kapture_to_hloc_image_list(kapture_path_map, image_list_map_path)
        feature_path_map = extract_features.main(feature_conf, Path(
            images_map), Path(output_dir), image_list=Path(image_list_map_path))
        assert feature_path_map.resolve() == feature_path.resolve()
    if "extract_features_query" not in skip_list:
        image_list_query_path = path.join(output_dir, 'image_list_query.txt')
        convert_kapture_to_hloc_image_list(kapture_path_query, image_list_query_path)
        feature_path_query = extract_features.main(feature_conf, Path(
            images_query), Path(output_dir), image_list=Path(image_list_query_path))
        assert feature_path_query.resolve() == feature_path.resolve()

    pairsfile_path_map_pathlib = Path(pairsfile_path_map)
    match_name_map = feature_conf['output'] + '_' + matcher_conf["output"] + f'_{pairsfile_path_map_pathlib.stem}'
    map_match_path = Path(output_dir, match_name_map+'.h5')
    if 'match_map_pairs' not in skip_list:
        map_match_path_actual = match_features.main(matcher_conf, pairsfile_path_map_pathlib,
                                                    feature_conf['output'], Path(output_dir))
        assert map_match_path_actual.resolve() == map_match_path.resolve()

    exported_mapping_path = path.join(output_dir, '3D-models/exported_from_kapture')
    if 'kapture_export_map_to_colmap' not in skip_list:
        export_colmap(kapture_path_map, path.join(exported_mapping_path, 'colmap.db'), exported_mapping_path,
                      force_overwrite_existing=True)
        # convert .txt to .bin
        run_model_converter(colmap_binary, exported_mapping_path, exported_mapping_path, 'BIN')

    triangulate_path = path.join(output_dir, 'sfm_' + feature_conf_str + '_' + matcher_conf_str)
    if 'triangulate' not in skip_list:
        triangulation.main(
            Path(triangulate_path),
            Path(exported_mapping_path),
            Path(images_map),
            pairsfile_path_map_pathlib,
            feature_path,
            map_match_path,
            colmap_binary)

    pairsfile_path_query_pathlib = Path(pairsfile_path_query)
    match_name_query = feature_conf['output'] + '_' + matcher_conf["output"] + f'_{pairsfile_path_query_pathlib.stem}'
    query_match_path = Path(output_dir, match_name_query+'.h5')
    if 'match_query_pairs' not in skip_list:
        query_match_path_actual = match_features.main(matcher_conf, pairsfile_path_query_pathlib,
                                                      feature_conf['output'], Path(output_dir))
        assert query_match_path_actual.resolve() == query_match_path.resolve()

    query_as_txt = path.join(output_dir, 'image_list_with_intrinsics.txt')
    export_image_list(kapture_path_query, query_as_txt)
    results_file = path.join(output_dir, f'results_{feature_conf_str}_{matcher_conf_str}.txt')
    if 'localize' not in skip_list:
        localize_sfm.main(
            Path(triangulate_path),
            Path(query_as_txt),
            pairsfile_path_query_pathlib,
            feature_path,
            query_match_path,
            Path(results_file),
            covisibility_clustering=covisibility_clustering)

    results_full = path.join(output_dir, f'results_{feature_conf_str}_{matcher_conf_str}_fullnames.txt')
    results_kapture = path.join(output_dir, f'results_{feature_conf_str}_{matcher_conf_str}_kapture')
    if 'convert_results' not in skip_list:
        convert_results_format(query_as_txt, results_file, results_full)
        convert_results_to_kapture(kapture_path_query, results_full, results_kapture)
    if 'evaluate' not in skip_list and path.isfile(get_csv_fullpath(kapture.Trajectories, kapture_path_query)):
        local_evaluate_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_evaluate.py')
        evaluate_args = ['-v', str(logger.level),
                         '-i', results_kapture,
                         '--labels', f'hloc_{feature_conf_str}_{matcher_conf_str}',
                         '-gt', kapture_path_query,
                         '-o', path.join(results_kapture, 'eval')]
        evaluate_args += ['--bins'] + bins_as_str
        evaluate_args.append('-f')
        run_python_command(local_evaluate_path, evaluate_args, python_binary)

    LTVL2020_output_path = path.join(output_dir, f'results_{feature_conf_str}_{matcher_conf_str}_LTVL2020_style.txt')
    if 'export_LTVL2020' not in skip_list:
        export_LTVL2020_script_name, export_LTVL2020_args = get_benchmark_format_command(
            benchmark_format_style,
            results_kapture,
            LTVL2020_output_path,
            True,
            logger
        )
        local_export_LTVL2020_path = path.join(pipeline_import_paths.HERE_PATH,
                                               f'../../kapture/tools/{export_LTVL2020_script_name}')
        run_python_command(local_export_LTVL2020_path, export_LTVL2020_args, python_binary)


def hloc_pipeline_from_kapture_dataset_get_parser():
    """
    get the argparse object for the kapture_hloc_pipeline_from_kapture_dataset.py command
    """
    parser = argparse.ArgumentParser(description=('create a Colmap model (map) from data specified in kapture format.'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--kapture-map', required=True,
                        help='path to the kapture map directory')
    parser.add_argument('--query', required=True,
                        help='input path to kapture mapping data root directory')
    parser.add_argument('--pairsfile-map', required=True,
                        help='input path to mapping pairs')
    parser.add_argument('--pairsfile-query', required=True,
                        help='input path to query pairs')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory.')
    parser.add_argument('--feature-conf',
                        default='superpoint_max', choices=list(extract_features.confs.keys()),
                        type=str,
                        help='features to use in hloc')
    parser.add_argument('--matcher-conf',
                        default='superglue', choices=list(match_features.confs.keys()),
                        type=str,
                        help='matcher to use in hloc')
    parser.add_argument('--covisibility-clustering', action='store_true', default=False, required=False,
                        help='use covisibility_clustering=True in hloc localize')
    parser.add_argument('--bins', nargs='+', default=["0.25 2", "0.5 5", "5 10"],
                        help='the desired positions/rotations thresholds for bins'
                        'format is string : position_threshold_in_m space rotation_threshold_in_degree')
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
    parser.add_argument('-s', '--skip', choices=['convert_pairsfile_map',
                                                 'convert_pairsfile_query',
                                                 'extract_features_map',
                                                 'extract_features_query',
                                                 'match_map_pairs',
                                                 'kapture_export_map_to_colmap',
                                                 'triangulate',
                                                 'match_query_pairs',
                                                 'localize',
                                                 'convert_results',
                                                 'evaluate',
                                                 'export_LTVL2020'],
                        nargs='+', default=[],
                        help='steps to skip')
    return parser


def hloc_pipeline_from_kapture_dataset_command_line():
    """
    Parse the command line arguments to build a colmap map and localize using the given kapture data.
    """
    parser = hloc_pipeline_from_kapture_dataset_get_parser()
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('kapture_hloc_pipeline_from_kapture_dataset.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    python_binary = args.python_binary
    if args.auto_python_binary:
        python_binary = sys.executable
        logger.debug(f'python_binary set to {python_binary}')
    hloc_pipeline_from_kapture_dataset(args.kapture_map,
                                       args.query,
                                       args.pairsfile_map,
                                       args.pairsfile_query,
                                       args.output,
                                       args.feature_conf,
                                       args.matcher_conf,
                                       args.covisibility_clustering,
                                       args.bins,
                                       args.benchmark_style,
                                       args.colmap_binary,
                                       python_binary,
                                       args.skip)


if __name__ == '__main__':
    hloc_pipeline_from_kapture_dataset_command_line()
