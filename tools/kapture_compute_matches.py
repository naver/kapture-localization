#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
from functools import lru_cache
from typing import Optional
import torch
from tqdm import tqdm
import os

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.matching import MatchPairNnTorch
from kapture_localization.utils.pairsfile import get_pairs_from_file

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_from_file, get_all_tar_handlers
from kapture.io.features import get_descriptors_fullpath, get_matches_fullpath
from kapture.io.features import image_descriptors_from_file
from kapture.io.features import matches_check_dir, image_matches_to_file
from kapture.io.tar import TarCollection
from kapture.utils.Collections import try_get_only_key_from_collection

logger = logging.getLogger('compute_matches')


@lru_cache(maxsize=50)
def load_descriptors(descriptors_type: str, input_path: str,
                     tar_handler: Optional[TarCollection],
                     image_name: str, dtype, dsize):
    """
    load a descriptor. this functions caches up to 50 descriptors

    :param descriptors_type: type of descriptors, name of the descriptors subfolder
    :param input_path: input path to kapture input root directory
    :param tar_handler: collection of preloaded tar archives
    :param image_name: name of the image
    :param dtype: dtype of the numpy array
    :param dsize: size of the numpy array
    """
    descriptors_path = get_descriptors_fullpath(descriptors_type, input_path, image_name, tar_handler)
    return image_descriptors_from_file(descriptors_path, dtype, dsize)


def compute_matches(input_path: str,
                    descriptors_type: Optional[str],
                    pairsfile_path: str,
                    overwrite_existing: bool = False):
    """
    compute matches from descriptors. images to match are selected from a pairsfile (csv with name1, name2, score)

    :param input_path: input path to kapture input root directory
    :type input_path: str
    :param descriptors_type: type of descriptors, name of the descriptors subfolder
    :param pairsfile_path: path to pairs file (csv with 3 fields, name1, name2, score)
    :type pairsfile_path: str
    """
    logger.info(f'compute_matches. loading input: {input_path}')
    with get_all_tar_handlers(input_path,
                              mode={kapture.Keypoints: 'r',
                                    kapture.Descriptors: 'r',
                                    kapture.GlobalFeatures: 'r',
                                    kapture.Matches: 'a'}) as tar_handlers:
        kdata = kapture_from_dir(input_path, pairsfile_path, skip_list=[kapture.GlobalFeatures,
                                                                        kapture.Observations,
                                                                        kapture.Points3d],
                                 tar_handlers=tar_handlers)
        image_pairs = get_pairs_from_file(pairsfile_path, kdata.records_camera, kdata.records_camera)
        compute_matches_from_loaded_data(input_path,
                                         tar_handlers,
                                         kdata,
                                         descriptors_type,
                                         image_pairs,
                                         overwrite_existing)


def compute_matches_from_loaded_data(input_path: str,
                                     tar_handlers: Optional[TarCollection],
                                     kdata: kapture.Kapture,
                                     descriptors_type: Optional[str],
                                     image_pairs: list,
                                     overwrite_existing: bool = False):
    assert kdata.sensors is not None
    assert kdata.records_camera is not None
    assert kdata.descriptors is not None
    os.umask(0o002)

    if descriptors_type is None:
        descriptors_type = try_get_only_key_from_collection(kdata.descriptors)
    assert descriptors_type is not None
    assert descriptors_type in kdata.descriptors
    keypoints_type = kdata.descriptors[descriptors_type].keypoints_type
    # assert kdata.descriptors[descriptors_type].metric_type == "L2"

    matcher = MatchPairNnTorch(use_cuda=torch.cuda.is_available())
    new_matches = kapture.Matches()

    logger.info('compute_matches. entering main loop...')
    hide_progress_bar = logger.getEffectiveLevel() > logging.INFO
    skip_count = 0
    for image_path1, image_path2 in tqdm(image_pairs, disable=hide_progress_bar):
        if image_path1 == image_path2:
            continue
        if image_path1 > image_path2:
            image_path1, image_path2 = image_path2, image_path1

        # skip existing matches
        if (not overwrite_existing) \
                and (kdata.matches is not None) \
                and keypoints_type in kdata.matches \
                and ((image_path1, image_path2) in kdata.matches[keypoints_type]):
            new_matches.add(image_path1, image_path2)
            skip_count += 1
            continue

        if image_path1 not in kdata.descriptors[descriptors_type] \
                or image_path2 not in kdata.descriptors[descriptors_type]:
            logger.warning('unable to find descriptors for image pair : '
                           '\n\t{} \n\t{}'.format(image_path1, image_path2))
            continue

        descriptor1 = load_descriptors(descriptors_type, input_path, tar_handlers,
                                       image_path1, kdata.descriptors[descriptors_type].dtype,
                                       kdata.descriptors[descriptors_type].dsize)
        descriptor2 = load_descriptors(descriptors_type, input_path, tar_handlers,
                                       image_path2, kdata.descriptors[descriptors_type].dtype,
                                       kdata.descriptors[descriptors_type].dsize)
        matches = matcher.match_descriptors(descriptor1, descriptor2)
        matches_path = get_matches_fullpath((image_path1, image_path2), keypoints_type, input_path, tar_handlers)
        image_matches_to_file(matches_path, matches)
        new_matches.add(image_path1, image_path2)

    if not overwrite_existing:
        logger.debug(f'{skip_count} pairs were skipped because the match file already existed')
    if not matches_check_dir(new_matches, keypoints_type, input_path, tar_handlers):
        logger.critical('matching ended successfully but not all files were saved')

    # update kapture matches
    if kdata.matches is None:
        kdata.matches = {}
    if keypoints_type not in kdata.matches:
        kdata.matches[keypoints_type] = kapture.Matches()
    kdata.matches[keypoints_type].update(new_matches)

    logger.info('all done')


def compute_matches_command_line():
    parser = argparse.ArgumentParser(
        description='Compute matches with nearest neighbors from descriptors.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help=('input path to kapture input root directory\n'
                              'it must contain all images (query + train) and their local features'))
    parser.add_argument('--pairsfile-path',
                        required=True,
                        type=str,
                        help=('text file in the csv format; where each line is image_name1, image_name2, score '
                              'which contains the image pairs to match'))
    parser.add_argument('-ow', '--overwrite', action='store_true', default=False,
                        help='overwrite matches if they already exist.')
    parser.add_argument('-desc', '--descriptors-type', default=None, help='kapture descriptors type.')
    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_matches(args.input,
                    args.descriptors_type,
                    args.pairsfile_path,
                    args.overwrite)


if __name__ == '__main__':
    compute_matches_command_line()
