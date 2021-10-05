# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import datetime

from kapture_localization.utils.logging import getLogger
import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture


def get_pairs_sequence(kdata: kapture.Kapture,
                       window_size: int,
                       loop: bool,
                       expand_window: bool,
                       max_interval: int):

    getLogger().info('computing pairs from sequences...')
    assert kdata.records_camera is not None
    start_total = datetime.datetime.now()
    out_of_bounds_images = set()  # records before or after trajectory

    localized_seq = {}
    getLogger().debug("Compute sequences")
    cur_seq = {}
    prev_ts = {}

    start_total = datetime.datetime.now()
    for timestamp, records in sorted(kdata.records_camera.items()):
        for sensor_id, image_name in records.items():
            if sensor_id not in localized_seq:
                localized_seq[sensor_id] = []
            if sensor_id not in cur_seq:
                cur_seq[sensor_id] = []
            if sensor_id not in prev_ts:
                prev_ts[sensor_id] = 0

            if prev_ts[sensor_id] == 0 or timestamp - prev_ts[sensor_id] > max_interval:
                # New sequence
                if len(cur_seq[sensor_id]) >= 2:
                    # image list
                    localized_seq[sensor_id].append(cur_seq[sensor_id])
                    getLogger().debug(
                        f'sequence {len(localized_seq[sensor_id])} '
                        f'for camera {sensor_id}, {len(cur_seq[sensor_id])} images')
                else:
                    for out_of_bound_img in cur_seq[sensor_id]:
                        out_of_bounds_images.add(out_of_bound_img)
                cur_seq[sensor_id] = []

            cur_seq[sensor_id].append(image_name)
            prev_ts[sensor_id] = timestamp

    for sensor_id, seq in cur_seq.items():
        if len(seq) >= 2:
            # ts list
            localized_seq[sensor_id].append(seq)
            getLogger().debug(
                f'sequence {len(localized_seq[sensor_id])} '
                f'for camera {sensor_id}, {len(seq)} images')
        else:
            for out_of_bound_img in seq:
                out_of_bounds_images.add(out_of_bound_img)

    image_pairs = {}
    for sensor_id, sequences in localized_seq.items():
        for sequence in sequences:
            for i in range(len(sequence)):
                if sequence[i] not in image_pairs:
                    image_pairs[sequence[i]] = []

                if loop:
                    cache = set()
                    for j in range(window_size):
                        right_index = (i+j+1) % len(sequence)
                        if sequence[right_index] != sequence[i] and sequence[right_index] not in cache:
                            image_pairs[sequence[i]].append((sequence[right_index], 1.0 - (j+1)/window_size))
                            cache.add(sequence[right_index])
                        left_index = (i-j-1) % len(sequence)
                        if sequence[left_index] != sequence[i] and sequence[left_index] not in cache:
                            image_pairs[sequence[i]].append((sequence[left_index], 1.0 - (j+1)/window_size))
                            cache.add(sequence[left_index])
                else:
                    range_on_right = min(len(sequence) - 1 - i, window_size)
                    range_on_left = min(i, window_size)

                    if expand_window:
                        final_range_on_right = range_on_right + (window_size - range_on_left)
                        final_range_on_left = range_on_left + (window_size - range_on_right)
                    else:
                        final_range_on_right = range_on_right
                        final_range_on_left = range_on_left
                    for j in range(final_range_on_right):
                        if i + j + 1 < len(sequence):
                            image_pairs[sequence[i]].append((sequence[i+j+1], 1.0 - (j+1)/final_range_on_right))
                    for j in range(final_range_on_left):
                        if i - j - 1 >= 0:
                            image_pairs[sequence[i]].append((sequence[i-j-1], 1.0 - (j+1)/final_range_on_left))

    images_pairs_list = []
    for query_image, pairs in sorted(image_pairs.items()):
        for map_image, score in sorted(pairs, key=lambda x: x[1], reverse=True):
            images_pairs_list.append([query_image, map_image, score])

    elapsed_total = datetime.datetime.now() - start_total
    getLogger().info(f'sequences processed in {elapsed_total.total_seconds()} seconds')
    getLogger().info(f'{len(out_of_bounds_images)} images were not in any sequence')
    getLogger().debug(f'list: {list(sorted(out_of_bounds_images))}')
    return images_pairs_list
