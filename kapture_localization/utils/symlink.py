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


def create_kapture_proxy(
        output_path: str,
        source_path: str,
        keypoints_path: Optional[str],
        descriptors_path: Optional[str],
        global_features_path: Optional[str],
        matches_path: Optional[str],
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
        absolute_symlink(keypoints_path, os.path.join(reconstruction_out_path, 'keypoints'))

    if descriptors_path is not None:
        assert path.exists(descriptors_path)
        absolute_symlink(descriptors_path, os.path.join(reconstruction_out_path, 'descriptors'))

    if global_features_path is not None:
        assert path.exists(global_features_path)
        absolute_symlink(global_features_path, os.path.join(reconstruction_out_path, 'global_features'))

    if matches_path is not None:
        assert path.exists(matches_path)
        absolute_symlink(matches_path, os.path.join(reconstruction_out_path, 'matches'))
