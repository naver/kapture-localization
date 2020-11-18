#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import os.path as path
import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils.paths import safe_remove_any_path

HERE_PATH = path.normpath(path.dirname(__file__))
tutorial_folder = path.join(HERE_PATH, 'tutorial')

if path.isdir(tutorial_folder):
    safe_remove_any_path(tutorial_folder, force=False)

matches_no_gv_folder = path.join(HERE_PATH, 'local_features/r2d2_500/NN_no_gv')
if path.isdir(matches_no_gv_folder):
    safe_remove_any_path(matches_no_gv_folder, force=False)

matches_colmap_gv_folder = path.join(HERE_PATH, 'local_features/r2d2_500/NN_colmap_gv')
if path.isdir(matches_colmap_gv_folder):
    safe_remove_any_path(matches_colmap_gv_folder, force=False)
