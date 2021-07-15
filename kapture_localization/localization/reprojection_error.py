# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import numpy as np
import quaternion
from typing import List
import cv2

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture


def compute_reprojection_error(pose: kapture.PoseTransform, num_inliers: int, inliers: List,
                               points2D, points3D, K, distortion):
    """
    compute reprojection error from a pose, a list of inlier indexes, the full list of 2D points and 3D points
    and camera parameters
    """
    obs_2D = np.empty((num_inliers, 2), dtype=np.float)
    obs_3D = np.empty((1, num_inliers, 3), dtype=np.float)
    for i, index in enumerate(inliers):
        obs_2D[i, :] = points2D[index]
        obs_3D[0, i, :] = points3D[index]

    rvec, _ = cv2.Rodrigues(quaternion.as_rotation_matrix(pose.r))
    tvec = pose.t
    estimated_points, _ = cv2.projectPoints(objectPoints=obs_3D,
                                            rvec=rvec, tvec=tvec, cameraMatrix=K, distCoeffs=distortion)
    estimated_points = estimated_points.reshape(obs_2D.shape)

    diff = estimated_points - obs_2D
    error = np.linalg.norm(diff, axis=1)
    residuals = np.sum(error)
    reprojection_error = residuals / num_inliers
    return reprojection_error
