# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import os.path as path
import ctypes
import sys
from typing import Optional, List

import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils.paths import safe_remove_any_path
from kapture.io.features import guess_feature_name_from_path


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
        keypoints_paths: Optional[List[Optional[str]]],
        descriptors_paths: Optional[List[str]],
        global_features_paths: Optional[List[str]],
        matches_paths: Optional[List[Optional[str]]],
        keypoints_types: Optional[List[Optional[str]]],
        descriptors_types: Optional[List[Optional[str]]],
        global_features_types: Optional[List[Optional[str]]],
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
    :param keypoints_paths: paths to keypoints root directory. Remapped to output_path/reconstruction/keypoints
    :param descriptors_paths: paths to descriptors root directory. Remapped to output_path/reconstruction/descriptors
    :param global_features_paths: paths to global features root directory.
                                 Remapped to output_path/reconstruction/global_features_path
    :param matches_paths: paths to matches root directory. Remapped to output_path/reconstruction/matches_path
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

    if keypoints_paths is not None and matches_paths is not None:
        assert len(keypoints_paths) == len(matches_paths)
    if keypoints_paths is not None and keypoints_types is not None:
        assert len(keypoints_paths) == len(keypoints_types)
    if matches_paths is not None and keypoints_types is not None:
        assert len(matches_paths) == len(keypoints_types)
    if descriptors_paths is not None and descriptors_types is not None:
        assert len(descriptors_paths) == len(descriptors_types)
    if global_features_paths is not None and global_features_types is not None:
        assert len(global_features_paths) == len(global_features_types)

    if keypoints_paths is not None:
        if keypoints_types is None:
            keypoints_types = [None for _ in range(len(keypoints_paths))]
        for i, keypoints_path in enumerate(keypoints_paths):
            if not keypoints_path:
                continue
            assert path.exists(keypoints_path)
            if keypoints_types[i] is None:
                keypoints_types[i] = guess_feature_name_from_path(keypoints_path)
            os.makedirs(os.path.join(reconstruction_out_path, 'keypoints'), exist_ok=True)
            absolute_symlink(keypoints_path,
                             os.path.join(reconstruction_out_path, 'keypoints', keypoints_types[i]))

    if descriptors_paths is not None:
        if descriptors_types is None:
            descriptors_types = [None for _ in range(len(descriptors_paths))]
        for i, descriptors_path in enumerate(descriptors_paths):
            assert path.exists(descriptors_path)
            if descriptors_types[i] is None:
                descriptors_types[i] = guess_feature_name_from_path(descriptors_path)
            os.makedirs(os.path.join(reconstruction_out_path, 'descriptors'), exist_ok=True)
            absolute_symlink(descriptors_path,
                             os.path.join(reconstruction_out_path, 'descriptors', descriptors_types[i]))

    if global_features_paths is not None:
        if global_features_types is None:
            global_features_types = [None for _ in range(len(global_features_paths))]
        for i, global_features_path in enumerate(global_features_paths):
            assert path.exists(global_features_path)
            if global_features_types[i] is None:
                global_features_types[i] = guess_feature_name_from_path(global_features_path)
            os.makedirs(os.path.join(reconstruction_out_path, 'global_features'), exist_ok=True)
            absolute_symlink(global_features_path,
                             os.path.join(reconstruction_out_path, 'global_features', global_features_types[i]))

    if matches_paths is not None:
        if keypoints_types is None:
            keypoints_types = [None for _ in range(len(matches_paths))]
        for i, matches_path in enumerate(matches_paths):
            if not matches_path:
                continue
            assert path.exists(matches_path)
            if keypoints_types[i] is None:
                keypoints_types[i] = guess_feature_name_from_path(matches_path)
            os.makedirs(os.path.join(reconstruction_out_path, 'matches'), exist_ok=True)
            absolute_symlink(matches_path, os.path.join(reconstruction_out_path, 'matches', keypoints_types[i]))


def create_kapture_proxy_single_features(
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
    create_kapture_proxy(output_path, source_path,
                         [keypoints_path] if keypoints_path else None,
                         [descriptors_path] if descriptors_path else None,
                         [global_features_path] if global_features_path else None,
                         [matches_path] if matches_path else None,
                         [keypoints_type] if keypoints_type else None,
                         [descriptors_type] if descriptors_type else None,
                         [global_features_type] if global_features_type else None,
                         force)
