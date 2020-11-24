#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import os.path as path
import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils.paths import safe_remove_any_path

HERE_PATH = path.normpath(path.dirname(__file__))
colmap_sfm_folder = path.join(HERE_PATH, 'colmap-sfm')
colmap_localization_folder = path.join(HERE_PATH, 'colmap-localization')
sift_colmap_vocab_tree_folder = path.join(HERE_PATH, 'sift_colmap_vocab_tree')
ir_bench_folder = path.join(HERE_PATH, 'image_retrieval_benchmark')

if path.isdir(colmap_sfm_folder):
    safe_remove_any_path(colmap_sfm_folder, force=False)
if path.isdir(colmap_localization_folder):
    safe_remove_any_path(colmap_localization_folder, force=False)
if path.isdir(sift_colmap_vocab_tree_folder):
    safe_remove_any_path(sift_colmap_vocab_tree_folder, force=False)
if path.isdir(ir_bench_folder):
    safe_remove_any_path(ir_bench_folder, force=False)

matches_no_gv_folder = path.join(HERE_PATH, 'local_features/r2d2_500/NN_no_gv')
if path.isdir(matches_no_gv_folder):
    safe_remove_any_path(matches_no_gv_folder, force=False)

matches_colmap_gv_folder = path.join(HERE_PATH, 'local_features/r2d2_500/NN_colmap_gv')
if path.isdir(matches_colmap_gv_folder):
    safe_remove_any_path(matches_colmap_gv_folder, force=False)
