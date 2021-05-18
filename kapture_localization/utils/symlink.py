# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import os.path as path
import ctypes
import sys
from typing import Optional

import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils.paths import safe_remove_any_path


def can_use_symlinks():
    """ Returns true if the current system is capable of symlinks."""
    if sys.platform.startswith("win"):
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False
    else:
        return True


def absolute_symlink(
        source_path: str,
        dest_path: str
):
    """ make sure it create an absolute symlink, even if source uis relative."""
    os.symlink(os.path.abspath(source_path), dest_path)


def guess_feature_name(feature_path: str) -> str:
    feature_path_c = os.path.abspath(feature_path).replace('\\', '/').rstrip('/')
    feature_path_split = feature_path_c.split('/')
    feature_name = None
    if feature_path_split[-1] not in ['keypoints', 'descriptors', 'global_features', 'matches']:
        feature_name = feature_path_split[-1]
    else:
        for parent_folder_name in ['local_features', 'global_features']:
            try:
                index = feature_path_split.index(parent_folder_name)
                if index + 1 < len(feature_path_split):
                    feature_name = feature_path_split[index + 1]
                    break
            except Exception:
                continue
    if feature_name is None:
        raise ValueError(f'failed to guess feature name from path {feature_path}')
    return feature_name


def create_kapture_proxy(
        output_path: str,
        source_path: str,
        keypoints_path: Optional[str],
        descriptors_path: Optional[str],
        global_features_path: Optional[str],
        matches_path: Optional[str],
        keypoints_type: Optional[str],
        descriptors_type: Optional[str],
        global_features_type: Optional[str],
        force: bool
):
    """
    Creates a kapture proxy directory based on  another one,
     and optionally gathering multiple source of reconstruction data.
    It heavily uses symlinks to minimize the amount of copied data.
    A source kapture directory is mandatory for the sensors part of kapture.
    All other reconstruction data, if given, are added.

    :param output_path: root path where to save the proxy. It will be cleaned if already exists.
    :param source_path: root path of the input kapture directory containing sensors.
    :param keypoints_path: path to keypoints root directory. Remapped to output_path/reconstruction/keypoints
    :param descriptors_path: path to descriptors root directory. Remapped to output_path/reconstruction/descriptors
    :param global_features_path: path to global features root directory.
                                 Remapped to output_path/reconstruction/global_features_path
    :param matches_path: path to matches root directory. Remapped to output_path/reconstruction/matches_path
    :param force: for to clean output (if needed) without user prompt.
    """
    if path.exists(output_path):
        safe_remove_any_path(output_path, force)
    assert not path.exists(output_path)
    os.makedirs(output_path)

    sensors_in_path = path.join(source_path, 'sensors')
    assert path.exists(sensors_in_path)

    sensors_out_path = path.join(output_path, 'sensors')
    absolute_symlink(sensors_in_path, sensors_out_path)

    reconstruction_out_path = path.join(output_path, 'reconstruction')
    os.makedirs(reconstruction_out_path)
    if keypoints_path is not None:
        assert path.exists(keypoints_path)
        if keypoints_type is None:
            keypoints_type = guess_feature_name(keypoints_path)
        os.makedirs(os.path.join(reconstruction_out_path, 'keypoints'), exist_ok=True)
        absolute_symlink(keypoints_path, os.path.join(reconstruction_out_path, 'keypoints', keypoints_type))

    if descriptors_path is not None:
        assert path.exists(descriptors_path)
        if descriptors_type is None:
            descriptors_type = guess_feature_name(descriptors_path)
        os.makedirs(os.path.join(reconstruction_out_path, 'descriptors'), exist_ok=True)
        absolute_symlink(descriptors_path, os.path.join(reconstruction_out_path, 'descriptors', descriptors_type))

    if global_features_path is not None:
        assert path.exists(global_features_path)
        if global_features_type is None:
            global_features_type = guess_feature_name(global_features_path)
        os.makedirs(os.path.join(reconstruction_out_path, 'global_features'), exist_ok=True)
        absolute_symlink(global_features_path, os.path.join(reconstruction_out_path,
                                                            'global_features',
                                                            global_features_type))

    if matches_path is not None:
        assert path.exists(matches_path)
        if keypoints_type is None:
            keypoints_type = guess_feature_name(matches_path)
        os.makedirs(os.path.join(reconstruction_out_path, 'matches'), exist_ok=True)
        absolute_symlink(matches_path, os.path.join(reconstruction_out_path, 'matches', keypoints_type))
