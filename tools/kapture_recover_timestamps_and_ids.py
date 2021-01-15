#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.csv as csv
import kapture.io.records
from kapture.io.structure import delete_existing_kapture_files

logger = logging.getLogger('recover_timestamps_and_ids')


def recover_timestamps_and_ids(input_path: str,
                               reference_path: str,
                               output_path: str,
                               image_transfer: kapture.io.records.TransferAction,
                               force: bool):
    """
    recover timestamps and sensor id from reference kapture data. useful after colmap export / import.
    output will only contains records that are in the reference. does not handle reconstruction data

    :param input_path: input path to kapture data root directory
    :type input_path: str
    :param reference_path: input path to kapture data reference root directory
    :type reference_path: str
    :param output_path: output directory
    :type output_path: str
    :param image_transfer: How to import images
    :type image_transfer: kapture.io.records.TransferAction
    :param force: force delete output directory if already exists
    :type force: bool
    """
    logger.info('recover_timestamps_and_ids...')
    os.makedirs(output_path, exist_ok=True)
    delete_existing_kapture_files(output_path, force_erase=force)

    logger.info('loading data ...')
    kdata = csv.kapture_from_dir(input_path, skip_list=[kapture.Keypoints,
                                                        kapture.Descriptors,
                                                        kapture.Matches,
                                                        kapture.GlobalFeatures,
                                                        kapture.Observations,
                                                        kapture.Points3d])
    kdata_ref = csv.kapture_from_dir(reference_path, skip_list=[kapture.Keypoints,
                                                                kapture.Descriptors,
                                                                kapture.Matches,
                                                                kapture.GlobalFeatures,
                                                                kapture.Observations,
                                                                kapture.Points3d])
    assert kdata.records_camera is not None
    assert kdata_ref.records_camera is not None

    logger.debug('reverse reference images')
    reference_images = {image_name: (timestamp, camera_id)
                        for timestamp, camera_id, image_name in kapture.flatten(kdata_ref.records_camera)}

    if kdata_ref.rigs is not None:
        reference_sensor_to_rig = {camera_id: rig_id
                                   for rig_id, camera_id, _ in kapture.flatten(kdata_ref.rigs)}
    else:
        reference_sensor_to_rig = {}

    if kdata.rigs is not None:
        sensor_to_rig = {camera_id: rig_id
                         for rig_id, camera_id, _ in kapture.flatten(kdata.rigs)}
    else:
        sensor_to_rig = {}

    # TODO lidar, wifi ...
    out_records = kapture.RecordsCamera()
    out_trajectories = kapture.Trajectories()
    sensor_mapping = {}

    logger.info('recover records and trajectories')
    for timestamp, camera_id, image_name in kapture.flatten(kdata.records_camera):
        if image_name not in reference_images:
            continue
        ref_timestamp, ref_camera_id = reference_images[image_name]
        sensor_mapping[camera_id] = ref_camera_id
        out_records[ref_timestamp, ref_camera_id] = image_name

        if (timestamp, camera_id) in kdata.trajectories:
            out_trajectories[ref_timestamp, ref_camera_id] = kdata.trajectories[timestamp, camera_id]
        if camera_id in sensor_to_rig and (timestamp, sensor_to_rig[camera_id]) in kdata.trajectories:
            assert ref_camera_id in reference_sensor_to_rig
            ref_rig_id = reference_sensor_to_rig[ref_camera_id]
            rig_id = sensor_to_rig[camera_id]
            out_trajectories[ref_timestamp, ref_rig_id] = kdata.trajectories[timestamp, rig_id]

    logger.info('recover sensor ids in sensors')
    out_sensors = kapture.Sensors({ref_camera_id: kdata.sensors[camera_id]
                                   for camera_id, ref_camera_id in sensor_mapping.items()})

    logger.info('recover rig ids in rigs')
    out_rigs = kapture.Rigs()
    for camera_id, ref_camera_id in sensor_mapping.items():
        if camera_id in sensor_to_rig and ref_camera_id in reference_sensor_to_rig:
            ref_rig_id = reference_sensor_to_rig[ref_camera_id]
            rig_id = sensor_to_rig[camera_id]
            out_rigs[ref_rig_id, ref_camera_id] = kdata[rig_id, camera_id]

    # prefer None over empty
    out_sensors = out_sensors or None
    out_records = out_records or None
    out_rigs = out_rigs or None
    out_trajectories = out_trajectories or None

    logger.info('saving results')
    kdata_out = kapture.Kapture(sensors=out_sensors, rigs=out_rigs,
                                trajectories=out_trajectories, records_camera=out_records)
    csv.kapture_to_dir(output_path, kdata_out)

    logger.info('handle image files with a call to transfer_actions')
    image_list_copy = {f for _, _, f in kapture.flatten(kdata_out.records_camera)}
    image_list_reference = {f for _, _, f in kapture.flatten(kdata.records_camera)}
    image_list = list(set.intersection(image_list_copy, image_list_reference))
    logger.info('import image files ...')
    image_dirpath_source = kapture.io.records.get_image_fullpath(input_path, None)
    if os.path.exists(image_dirpath_source):
        kapture.io.records.import_record_data_from_dir_auto(image_dirpath_source, output_path,
                                                            image_list, image_transfer)
    logger.info('done.')


def recover_timestamps_and_ids_command_line():
    parser = argparse.ArgumentParser(description=('recover timestamps and sensor id from reference kapture data.'
                                                  ' useful after colmap export / import.'
                                                  ' output will only contains records that are in the reference.'
                                                  ' does not handle reconstruction data'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-ref', '--reference', required=True,
                        help=('input path to kapture data reference root directory'
                              ' if it does not contain all records, '
                              'then only the corresponding subset of input will be written to output'))
    parser.add_argument('-o', '--output', required=False,
                        help='output directory')
    parser.add_argument('--image_transfer', type=kapture.io.records.TransferAction,
                        default=kapture.io.records.TransferAction.skip,
                        help=(f'How to import images [skip], '
                              f'choose among: {", ".join(a.name for a in kapture.io.records.TransferAction)}'))
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output directory if already exists')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    logger.debug('kapture_recover_timestamps_and_ids.py \\\n' +
                 ''.join(['\n\t{:13} = {}'.format(k, v) for k, v in vars(args).items()]))
    recover_timestamps_and_ids(args.input, args.reference, args.output, args.image_transfer, args.force)


if __name__ == '__main__':
    recover_timestamps_and_ids_command_line()
