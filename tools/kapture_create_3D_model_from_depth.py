#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import math
from tqdm import tqdm
import numpy as np
from PIL import Image
from typing import List, Optional
import cv2
from enum import auto
from functools import lru_cache

import path_to_kapture_localization  # noqa: F401
import kapture_localization.utils.logging
from kapture_localization.utils.cv_camera_matrix import get_camera_matrix_from_kapture

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, kapture_to_dir, get_all_tar_handlers
from kapture.io.records import depth_map_from_file, get_image_fullpath, get_depth_map_fullpath
from kapture.io.tar import TarCollection
from kapture.utils.Collections import try_get_only_key_from_collection
from kapture.utils import AutoEnum
from kapture.io.features import get_keypoints_fullpath, image_keypoints_from_file

logger = logging.getLogger("create_3D_model_from_depth")


@lru_cache(maxsize=50)
def load_keypoints(keypoints_type, input_path, image_name, dtype, dsize, tar_handlers):
    keypoints_path = get_keypoints_fullpath(keypoints_type, input_path, image_name, tar_handlers)
    return image_keypoints_from_file(keypoints_path, dtype, dsize)


class Method(AutoEnum):
    voxelgrid = auto()
    all = auto()

    def __str__(self):
        return self.value


class VoxelGrid:
    def __init__(self, cellsizes):
        self.cellsizes = list(map(float, cellsizes))
        self.levels = len(self.cellsizes)
        self.min_cellsize = self.cellsizes[self.levels - 1]
        self.cells = {}
        self.grids = {}
        self.count = self.levels * [0]

    def get_voxelgrid_cell(self, pt, cellsize):
        x = math.floor(pt[0] / cellsize)
        y = math.floor(pt[1] / cellsize)
        z = math.floor(pt[2] / cellsize)

        return (x, y, z)

    def append(self, pt, img_name):
        idx = self.get_voxelgrid_cell(pt, self.min_cellsize)
        if img_name not in self.cells[idx][1]:
            self.cells[idx][1].append(img_name)
            return self.cells[idx]
        return None

    def add(self, pt, pt_idx, img_name):
        idx = self.get_voxelgrid_cell(pt, self.min_cellsize)
        self.cells[idx] = (pt_idx, [img_name])

    def create_indices(self, pt):
        idx = []
        for i in range(0, self.levels):
            idx.append(self.get_voxelgrid_cell(pt, self.cellsizes[i]))
        return idx

    def set(self, grids, idx, count):
        if count == self.levels - 1:
            if not idx[count] in grids:
                grids[idx[count]] = 1
                self.count[count] += 1
                return False
            else:
                return True
        else:
            if not idx[count] in grids:
                grids[idx[count]] = {}
                self.count[count] += 1
            return self.set(grids[idx[count]], idx, count + 1)

    def exists(self, pt):
        idx = self.create_indices(pt)
        return self.set(self.grids, idx, 0)

    def print(self):
        str = ''
        for i, c in enumerate(self.count):
            str += f'{i}: {c} '
        logger.info(str)


def project_kp_to_3D(u, v, d, cx, cy, fx, fy):
    x = d * ((u - cx) / fx)
    y = d * ((v - cy) / fy)
    return (x, y, d)


def create_3D_model_from_depth(input_path: str,
                               output_path: str,
                               keypoints_type: Optional[str],
                               depth_sensor_id: str,
                               topk: int,
                               method: Method,
                               cellsizes: List[str],
                               force: bool):
    """
    Create 3D model from a kapture dataset that has registered depth data
    Loads the kapture data then call create_3D_model_from_depth_from_loaded_data
    """
    if os.path.exists(output_path) and not force:
        print(f'outpath already exists, use --force to overwrite')
        return -1

    logger.info(f'loading {input_path}')
    with get_all_tar_handlers(input_path,
                              mode={kapture.Keypoints: 'r',
                                    kapture.Descriptors: 'r',
                                    kapture.GlobalFeatures: 'r',
                                    kapture.Matches: 'a'}) as tar_handlers:
        kdata = kapture_from_dir(input_path, tar_handlers=tar_handlers)
        create_3D_model_from_depth_from_loaded_data(kdata, input_path, tar_handlers,
                                                    output_path, keypoints_type,
                                                    depth_sensor_id, topk,
                                                    method, cellsizes, force)


def create_3D_model_from_depth_from_loaded_data(kdata: kapture.Kapture,
                                                input_path: str,
                                                tar_handlers: TarCollection,
                                                output_path: str,
                                                keypoints_type: Optional[str],
                                                depth_sensor_id: str,
                                                topk: int,
                                                method: Method,
                                                cellsizes: List[str],
                                                force: bool):
    """
    Create 3D model from a kapture dataset that has registered depth data
    Assumes the kapture data is already loaded
    """
    logger.info(f'create 3D model using depth data')

    if os.path.exists(output_path) and not force:
        print(f'outpath already exists, use --force to overwrite')
        return -1

    if kdata.rigs is not None:
        assert kdata.trajectories is not None
        kapture.rigs_remove_inplace(kdata.trajectories, kdata.rigs)

    if keypoints_type is None:
        keypoints_type = try_get_only_key_from_collection(kdata.keypoints)
    assert keypoints_type is not None
    assert kdata.keypoints is not None
    assert keypoints_type in kdata.keypoints

    if method == Method.voxelgrid:
        vg = VoxelGrid(cellsizes)

    # add all 3D points to map that correspond to a keypoint
    logger.info('adding points from scan to kapture')
    points3d = []
    observations = kapture.Observations()

    progress_bar = tqdm(total=len(list(kapture.flatten(kdata.records_camera, is_sorted=True))),
                        disable=logger.level >= logging.CRITICAL)
    for timestamp, sensor_id, sensing_filepath in kapture.flatten(kdata.records_camera, is_sorted=True):
        logger.info(f'total 3d points: {len(points3d)}, processing {sensing_filepath}')
        # check if images have a pose
        if timestamp not in kdata.trajectories:
            logger.info('{} does not have a pose. skipping ...'.format(sensing_filepath))
            continue

        # check if depth map exists
        depth_map_record = ''
        if timestamp in kdata.records_depth:
            if depth_sensor_id is None:
                depth_id = sensor_id + '_depth'
            else:
                depth_id = depth_sensor_id
            if depth_id in kdata.records_depth[timestamp]:
                depth_map_record = kdata.records_depth[timestamp][depth_id]
        depth_map_size = tuple([int(x) for x in kdata.sensors[depth_id].camera_params[0:2]])
        depth_path = get_depth_map_fullpath(input_path, depth_map_record)
        if not os.path.exists(depth_path):
            logger.info('no 3D data found for {}. skipping ...'.format(sensing_filepath))
            continue
        depth_map = depth_map_from_file(depth_path, depth_map_size)
        img = Image.open(get_image_fullpath(input_path, sensing_filepath)).convert(
            'RGB')

        assert img.size[0] == depth_map_size[0]
        assert img.size[1] == depth_map_size[1]

        kps_raw = load_keypoints(keypoints_type, input_path,
                                 sensing_filepath,
                                 kdata.keypoints[keypoints_type].dtype,
                                 kdata.keypoints[keypoints_type].dsize,
                                 tar_handlers)

        _, camera_sensor_C, camera_dist = get_camera_matrix_from_kapture(np.zeros((1, 0, 2), dtype=np.float64),
                                                                         kdata.sensors[sensor_id])
        cv2_keypoints, depth_sensor_C, depth_dist = get_camera_matrix_from_kapture(kps_raw, kdata.sensors[depth_id])
        assert np.isclose(depth_sensor_C, camera_sensor_C).all()
        assert np.isclose(depth_dist, camera_dist).all()

        if np.count_nonzero(camera_dist) > 0:
            epsilon = np.finfo(np.float64).eps
            stop_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, epsilon)
            undistorted_cv2_keypoints = cv2.undistortPointsIter(cv2_keypoints, camera_sensor_C, camera_dist,
                                                                R=None, P=camera_sensor_C,
                                                                criteria=stop_criteria)
        else:
            undistorted_cv2_keypoints = cv2_keypoints

        cv2_keypoints = cv2_keypoints.reshape((kps_raw.shape[0], 2))
        undistorted_cv2_keypoints = undistorted_cv2_keypoints.reshape((kps_raw.shape[0], 2))

        points3d_img = []
        rgb_img = []
        kp_idxs = []
        for idx_kp, kp in enumerate(cv2_keypoints[0:topk]):
            u = round(kp[0])
            v = round(kp[1])

            undist_kp = undistorted_cv2_keypoints[idx_kp]
            undist_u = round(undist_kp[0])
            undist_v = round(undist_kp[1])

            if u >= 0 and u < depth_map_size[0] and v >= 0 and v < depth_map_size[1]:
                if depth_map[v, u] == 0:
                    continue
                pt3d = project_kp_to_3D(undist_u, undist_v, depth_map[v, u],
                                        depth_sensor_C[0, 2], depth_sensor_C[1, 2],
                                        depth_sensor_C[0, 0], depth_sensor_C[1, 1])
                points3d_img.append(pt3d)
                rgb_img.append(img.getpixel((u, v)))
                kp_idxs.append(idx_kp)
        # transform to world coordinates (pt3d from a depth map is in camera coordinates)
        # we use sensor_id here because we assume that the image and the corresponding depthmap have the same pose
        # and sometimes, the pose might only be provided for the images
        cam_to_world = kdata.trajectories[timestamp][sensor_id].inverse()
        if len(points3d_img) == 0:
            continue
        points3d_img = cam_to_world.transform_points(np.array(points3d_img))
        for idx_kp, pt3d, rgb in zip(kp_idxs, points3d_img, rgb_img):
            if not np.isnan(pt3d).any():
                # apply transform (alignment)
                if method == Method.voxelgrid:
                    assert vg is not None
                    if not vg.exists(pt3d):
                        # add 3D point
                        points3d.append(list(pt3d) + list(rgb))
                        # add observation
                        observations.add(len(points3d) - 1, keypoints_type, sensing_filepath, idx_kp)
                        vg.add(pt3d, len(points3d) - 1, sensing_filepath)
                    else:
                        ret = vg.append(pt3d, sensing_filepath)
                        if ret is not None:
                            observations.add(ret[0], keypoints_type, sensing_filepath, idx_kp)
                elif method == Method.all:
                    # add 3D point
                    points3d.append(list(pt3d) + list(rgb))
                    # add observation
                    observations.add(len(points3d) - 1, keypoints_type, sensing_filepath, idx_kp)
        # save_3Dpts_to_ply(points3d, os.path.join(output_path, 'map.ply'))
        progress_bar.update(1)
    progress_bar.close()

    kdata.points3d = kapture.Points3d(np.array(points3d))
    kdata.observations = observations

    logger.info('saving ...')
    kapture_to_dir(output_path, kdata)
    # save_3Dpts_to_ply(points3d, os.path.join(output_path, 'map.ply'))

    logger.info('all done')


def create_3D_model_from_depth_command_line():
    """
    build the argparse for create_3D_model_from_depth then call it
    """
    parser = argparse.ArgumentParser(
        description='Create 3D model from a kapture dataset that has registered depth data.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help=('input path to kapture dataset'))
    parser.add_argument('-o', '--output', required=True,
                        help=('output path to kapture dataset'))
    parser.add_argument('--keypoints-type', default=None, help='kapture keypoints type.')
    parser.add_argument('-d', '--depth', default=None,
                        help=('depth sensor kapture id: if None, '
                              'then camera_id + _depth will be used; such as ipad0_depth'))
    parser.add_argument('-k', '--topk', required=False, default=20000, type=int,
                        help=('number of keypoints to use.'))
    parser.add_argument('--cellsizes', nargs='+', default=["10", "1", "0.01"],
                        help='cell sizes for hierarchical search')
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output directory if already exists')
    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)
    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    create_3D_model_from_depth(args.input, args.output, args.keypoints_type,
                               args.depth, args.topk,
                               Method.voxelgrid, args.cellsizes, args.force)


if __name__ == '__main__':
    create_3D_model_from_depth_command_line()
