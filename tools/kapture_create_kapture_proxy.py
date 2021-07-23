#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
from typing import List, Optional

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.symlink import can_use_symlinks, create_kapture_proxy
import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging

logger = logging.getLogger('create_kapture_proxy')


def _convert_none_string(line: str):
    if line.lower() == 'none':
        return None
    return line


def _convert_none_string_array(lines: Optional[List[str]]):
    if lines is None:
        return None
    return [_convert_none_string(line) for line in lines]


def create_kapture_proxy_command_line():
    parser = argparse.ArgumentParser(
        description='Create a proxy kapture from a source kapture with only sensors data and orphan features.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)

    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='silently output folder content if it already exist.')

    parser.add_argument('-i', '--input', required=True, help=('input path to kapture input root directory'
                                                              ' (only sensors will be used)'))
    parser.add_argument('-o', '--output', required=True, help='output path to the proxy kapture')

    parser.add_argument('-kpt', '--keypoints-path', default=None, nargs='+',
                        help='input path to the orphan keypoints folder')
    parser.add_argument('-desc', '--descriptors-path', default=None, nargs='+',
                        help='input path to the orphan descriptors folder')
    parser.add_argument('-gfeat', '--global-features-path', default=None, nargs='+',
                        help='input path to the orphan global features folder')
    parser.add_argument('-matches', '--matches-path', default=None, nargs='+',
                        help=('input path to the orphan matches folder, '
                              'if both keypoints-path and matches-paths are given, '
                              'the order of the two list must be the same (same as keypoints-type), '
                              'use the none if necessary, it will be converted to None in code'))

    parser.add_argument('--keypoints-type', default=None, nargs='+', help='kapture keypoints types.')
    parser.add_argument('--descriptors-type', default=None, nargs='+', help='kapture descriptors types.')
    parser.add_argument('--global-features-type', default=None, nargs='+', help='kapture global features types.')

    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))

    if can_use_symlinks():
        keypoints_paths = _convert_none_string_array(args.keypoints_path)
        descriptors_paths = args.descriptors_path
        global_features_path = args.global_features_path
        matches_paths = _convert_none_string_array(args.matches_path)

        keypoints_types = _convert_none_string_array(args.keypoints_type)
        descriptors_types = _convert_none_string_array(args.descriptors_type)
        global_features_types = _convert_none_string_array(args.global_features_type)

        create_kapture_proxy(args.output, args.input,
                             keypoints_paths, descriptors_paths,
                             global_features_path, matches_paths,
                             keypoints_types, descriptors_types,
                             global_features_types,
                             args.force)
    else:
        raise EnvironmentError('Please restart this command as admin, it is required for os.symlink'
                               'see https://docs.python.org/3.6/library/os.html#os.symlink')
        # need to find a way to redirect output, else it closes on error...
        # logger.critical('Request UAC for symlink rights...')
        # ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)


if __name__ == '__main__':
    create_kapture_proxy_command_line()
