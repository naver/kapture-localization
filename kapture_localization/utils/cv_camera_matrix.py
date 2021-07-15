# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import cv2
import numpy as np

from kapture_localization.utils.logging import getLogger

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture


def get_camera_matrix(fx: float, fy: float, cx: float, cy: float):
    """
    get numpy array for the intrinsic matrix
    """
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def _get_zero_distortion():
    return np.array([], dtype=np.float64)


def _get_distortion_4(k1, k2, p1, p2):
    return np.array([k1, k2, p1, p2], dtype=np.float64)


# def _get_distortion_5(k1, k2, p1, p2, k3):
#     return np.array([k1, k2, p1, p2, k3], dtype=np.float64)


def _get_distortion_8(k1, k2, p1, p2, k3, k4, k5, k6):
    return np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)


def is_model_opencv_compatible(sensor: kapture.Camera):
    """
    return 0 if the model is fully compatible with opencv
           1 if the model is compatible with opencv after undistorting points
           -1 if the model is not compatible with opencv
    """
    model = sensor.camera_type
    if model == kapture.CameraType.SIMPLE_PINHOLE or \
            model == kapture.CameraType.PINHOLE or \
            model == kapture.CameraType.SIMPLE_RADIAL or \
            model == kapture.CameraType.SIMPLE_RADIAL or \
            model == kapture.CameraType.OPENCV or \
            model == kapture.CameraType.FULL_OPENCV:
        return 0
    elif model == kapture.CameraType.OPENCV_FISHEYE or \
        model == kapture.CameraType.RADIAL_FISHEYE or \
            model == kapture.CameraType.SIMPLE_RADIAL_FISHEYE:
        return 1
    else:
        return -1


def get_camera_matrix_from_kapture(pts_2D, sensor: kapture.Camera):
    """
    returned keypoints of shape [1, number of keypoints, dsize]
    :param pts_2D: [description]
    :type pts_2D: keypoints in shape [number of keypoints,dsize] or [1, number of keypoints, dsize]
    """
    if len(pts_2D.shape) == 2:
        pts_2D = np.array([pts_2D], dtype=np.float64)
    assert len(pts_2D.shape) == 3
    pts_2D = pts_2D[:, :, 0:2]

    model = sensor.camera_type
    model_params = sensor.camera_params

    if model == kapture.CameraType.SIMPLE_PINHOLE:
        # w, h, f, cx, cy
        return (pts_2D,
                get_camera_matrix(model_params[2], model_params[2], model_params[3], model_params[4]),
                _get_zero_distortion())
    elif model == kapture.CameraType.PINHOLE:
        # w, h, fx, fy, cx, cy
        return (pts_2D,
                get_camera_matrix(model_params[2], model_params[3], model_params[4], model_params[5]),
                _get_zero_distortion())
    elif model == kapture.CameraType.SIMPLE_RADIAL:
        # w, h, f, cx, cy, k
        return (pts_2D,
                get_camera_matrix(model_params[2], model_params[2], model_params[3], model_params[4]),
                _get_distortion_4(model_params[5], 0, 0, 0))
    elif model == kapture.CameraType.RADIAL:
        # w, h, f, cx, cy, k1, k2
        return (pts_2D,
                get_camera_matrix(model_params[2], model_params[2], model_params[3], model_params[4]),
                _get_distortion_4(model_params[5], model_params[6], 0, 0))
    elif model == kapture.CameraType.OPENCV:
        # w, h, fx, fy, cx, cy, k1, k2, p1, p2
        return (pts_2D,
                get_camera_matrix(model_params[2], model_params[3], model_params[4], model_params[5]),
                _get_distortion_4(model_params[6], model_params[7], model_params[8], model_params[9]))
    elif model == kapture.CameraType.FULL_OPENCV:
        # w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        return (pts_2D,
                get_camera_matrix(model_params[2], model_params[3], model_params[4], model_params[5]),
                _get_distortion_8(model_params[6], model_params[7], model_params[8], model_params[9],
                                  model_params[10], model_params[11], model_params[12], model_params[13]))
    elif model == kapture.CameraType.OPENCV_FISHEYE:
        # w, h, fx, fy, cx, cy, k1, k2, k3, k4
        fisheye_camera_matrix = get_camera_matrix(model_params[2], model_params[3], model_params[4], model_params[5])
        fisheye_distortion = _get_distortion_4(model_params[6], model_params[7], model_params[8], model_params[9])
        P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(fisheye_camera_matrix, fisheye_distortion,
                                                                   (int(model_params[0]), int(model_params[1])), None)
        undist_pts_2D = cv2.fisheye.undistortPoints(pts_2D, fisheye_camera_matrix, fisheye_distortion, P=P)
        return undist_pts_2D, P, _get_zero_distortion()
    elif model == kapture.CameraType.RADIAL_FISHEYE or model == kapture.CameraType.SIMPLE_RADIAL_FISHEYE:
        getLogger().warning('OpenCV radial fisheye model not fully supported, distortion coefficients will be ignored')
        # w, h, f, cx, cy, k or w, h, f, cx, cy, k1, k2
        fisheye_camera_matrix = get_camera_matrix(model_params[2], model_params[2], model_params[3], model_params[4])
        fisheye_distortion = _get_distortion_4(0, 0, 0, 0)
        P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(fisheye_camera_matrix, fisheye_distortion,
                                                                   (int(model_params[0]), int(model_params[1])), None)
        undist_pts_2D = cv2.fisheye.undistortPoints(pts_2D, fisheye_camera_matrix, fisheye_distortion, P=P)
        return undist_pts_2D, P, _get_zero_distortion()
    else:
        raise ValueError(f'Camera model {model.value} not supported')


def opencv_model_to_kapture(width, height, K, distortion):
    """
    get kapture.Camera from opencv intrinsic matrix and distortion parameters
    """
    # opencv: k1, k2, p1, p2, k3, k4, k5, k6
    distortion = np.pad(distortion, [0, 8-len(distortion)], mode='constant', constant_values=0)

    # kapture: w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    params = [width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + list(distortion)
    return kapture.Camera(kapture.CameraType.FULL_OPENCV, params)
