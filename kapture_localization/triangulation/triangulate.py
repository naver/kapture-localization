# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import numpy as np
from numba import njit
from typing import List, Tuple


@njit(fastmath=True, cache=True)
def _homogeneous_to_euclidean(vector):
    # a vector expression of the N-1 first coefficients of *this divided by that last coefficient.
    n = vector.shape[0]
    return vector[0:(n-1)] / vector[n-1]


@njit(fastmath=True, cache=True)
def _triangulate_n_views_algebraic(views, projection_matrices):
    # see NOTICE for details on LICENSE
    # implementation inspired from
    # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/base/triangulation.cc#L72
    # https://github.com/opencv/opencv_contrib/blob/c99d1c3b0446037045acd0bf54bcbe148e721b5e/modules/sfm/src/triangulation.cpp#L96
    # https://github.com/openMVG/openMVG/blob/160643be515007580086650f2ae7f1a42d32e9fb/src/openMVG/multiview/triangulation_nview.cpp#L39
    number_of_views = views.shape[0]
    A = np.zeros((4, 4), dtype=np.float64)
    for i in range(number_of_views):
        point_norm = np.array([views[i, 0], views[i, 1], 1.0], dtype=np.float64)
        point_norm = point_norm / np.linalg.norm(point_norm)
        point = np.zeros((3, 1), dtype=np.float64)
        point[:, 0] = point_norm
        term = projection_matrices[i] - np.dot(point, np.dot(np.transpose(point), projection_matrices[i]))
        A = A + np.dot(np.transpose(term), term)

    _, v = np.linalg.eigh(A)
    return True, _homogeneous_to_euclidean(v[:, 0])


@njit(fastmath=True, cache=True)
def _get_residual(point3d,
                  pts2D,
                  projection_matrix):
    number_of_points = pts2D.shape[0]
    # use of empty array of fixed size instead of list is critical for speed
    residuals = np.full((number_of_points,), np.finfo(np.float64).max, dtype=np.float64)

    # define variables for numba
    x = np.float64(0)
    y = np.float64(0)
    z = np.float64(0)
    dx = np.float64(0)
    dy = np.float64(0)

    for i in range(0, number_of_points):
        x = point3d[0] * projection_matrix[i, 0, 0] + point3d[1] * projection_matrix[i, 0, 1] + \
            point3d[2] * projection_matrix[i, 0, 2] + projection_matrix[i, 0, 3]
        y = point3d[0] * projection_matrix[i, 1, 0] + point3d[1] * projection_matrix[i, 1, 1] + \
            point3d[2] * projection_matrix[i, 1, 2] + projection_matrix[i, 1, 3]
        z = point3d[0] * projection_matrix[i, 2, 0] + point3d[1] * projection_matrix[i, 2, 1] + \
            point3d[2] * projection_matrix[i, 2, 2] + projection_matrix[i, 2, 3]

        # Check if 3D point is in front of camera.
        if z > np.finfo(np.float64).eps:
            z = 1.0/z
            dx = pts2D[i, 0] - x * z
            dy = pts2D[i, 1] - y * z
            residuals[i] = dx * dx + dy * dy
    return residuals


@njit(fastmath=True, cache=True)
def _evaluate(residuals: List[float], max_residual: float) -> Tuple[int, float]:
    num_inliers = 0
    residual_sum = 0
    for residual in residuals:
        if residual <= max_residual:
            num_inliers += 1
            residual_sum += residual
    return num_inliers, residual_sum


@njit(cache=True)
def get_inliers(num_inliers, residuals, max_residual):
    """
    get the indexes of the inliers in residuals
    """
    inliers = np.empty((num_inliers,), dtype=np.int64)
    j = 0
    for i in range(residuals.shape[0]):
        if residuals[i] <= max_residual:
            inliers[j] = i
            j += 1
    return inliers


@njit(cache=True)
def triangulate_n_views_ransac(views, projection_matrices,
                               min_required_inliers, sample_count,
                               max_residual, num_iteration, combinations):
    """
    triangulate a 3d point from multiple views

    :param views: numpy array of undistorted kpts
    :param projection_matrices: numpy array of 3x4 projection_matrices (same len as views)
    :param min_required_inliers: minimum number of inlier views to be a valid 3d point
    :param sample_count: number of views used when triangulating for one iteration
    :param max_residual: maximum reprojection error to be a inlier
    :param num_iteration: maximum number of iterations
    :param combinations: numpy array of selected samples for each iteration. or empty array for random
    :return: tuple(is_success, point3d, num_inliers, residuals)
    """
    number_of_views = views.shape[0]
    best_model = np.array([0, 0, 0], dtype=np.float64)
    best_residual_sum = 1.7976931348623158e+308  # max float64
    best_num_inliers = 0
    best_residuals = np.empty((number_of_views,), dtype=np.float64)

    if number_of_views < min_required_inliers:
        return False, best_model, best_num_inliers, best_residuals

    indexes = np.arange(number_of_views)
    # define variables for numba
    use_combination = combinations.shape[0] == num_iteration
    sampled_indexes = np.empty((sample_count,), dtype=np.int64)
    min_set_views = np.empty((sample_count, 2), dtype=np.float64)
    min_set_projection_matrices = np.empty((sample_count, 3, 4), dtype=np.float64)

    success = False
    point3d = np.array([0, 0, 0], dtype=np.float64)
    residuals = np.empty((number_of_views,), dtype=np.float64)
    num_inliers = 0
    residual_sum = np.float64(0)
    for i in range(num_iteration):
        if use_combination:
            sampled_indexes = combinations[i]
        else:
            sampled_indexes = np.random.choice(indexes, size=sample_count, replace=False)
        min_set_views = views[sampled_indexes]
        min_set_projection_matrices = projection_matrices[sampled_indexes]

        success, point3d = _triangulate_n_views_algebraic(min_set_views, min_set_projection_matrices)
        if success:
            residuals = _get_residual(point3d, views, projection_matrices)
            num_inliers, residual_sum = _evaluate(residuals, max_residual)
            if num_inliers > best_num_inliers or \
                    (num_inliers == best_num_inliers and residual_sum < best_residual_sum):
                best_num_inliers = num_inliers
                best_residual_sum = residual_sum
                best_model = point3d
                best_residuals = residuals

    success = (best_residual_sum != 1.7976931348623158e+308) and best_num_inliers >= min_required_inliers
    return success, best_model, best_num_inliers, best_residuals
