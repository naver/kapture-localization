from packaging import version
try:
    import pycolmap
    version_number = pycolmap.__version__
    if version.parse(version_number) < version.parse("0.5.0"):
        compatibility_mode = True  # compatibility with older versions
    else:
        compatibility_mode = False
    has_pycolmap = True
except Exception as e:  # ModuleNotFoundError:
    print(e)
    has_pycolmap = False
    compatibility_mode = False
import numpy as np
import quaternion


def absolute_pose_estimation(points2D, points3D, camera_dict, max_error, min_inlier_ratio, min_num_iterations,
                             max_num_iterations, confidence):
    assert has_pycolmap
    if compatibility_mode:
        return pycolmap.absolute_pose_estimation(points2D, points3D, camera_dict, max_error,
                                                 min_inlier_ratio, min_num_iterations, max_num_iterations, confidence)
    else:
        pycolmap_camera = pycolmap.Camera(
            model=camera_dict['model'], width=camera_dict['width'], height=camera_dict['height'],
            params=camera_dict['params'])

        pycolmap_estimation_options = dict(ransac=dict(max_error=max_error, min_inlier_ratio=min_inlier_ratio,
                                           min_num_trials=min_num_iterations, max_num_trials=max_num_iterations,
                                           confidence=confidence))
        ret = pycolmap.absolute_pose_estimation(points2D, points3D, pycolmap_camera,
                                                estimation_options=pycolmap_estimation_options)
        if ret is None:
            ret = {'success': False}
        else:
            ret['success'] = True
            if callable(ret['cam_from_world'].matrix):
                retmat = ret['cam_from_world'].matrix()
            else:
                retmat = ret['cam_from_world'].matrix
            ret['qvec'] = quaternion.from_rotation_matrix(retmat[:3, :3])
            ret['tvec'] = retmat[:3, 3]
        return ret


def rig_absolute_pose_estimation(points2D, points3D, cameras_dict, qvec, tvec, max_error,
                                 min_inlier_ratio, min_num_iterations, max_num_iterations,
                                 confidence):
    assert has_pycolmap
    if compatibility_mode:
        return pycolmap.rig_absolute_pose_estimation(points2D, points3D, cameras_dict, qvec, tvec, max_error,
                                                     min_inlier_ratio, min_num_iterations, max_num_iterations,
                                                     confidence)
    else:
        pycolmap_cameras = []
        camera_idxs = []
        cams_from_rig = []
        for idx, (camera_dict, qvec_idx, tvec_idx) in enumerate(zip(cameras_dict, qvec, tvec)):
            pycolmap_cameras.append(pycolmap.Camera(
                                    model=camera_dict['model'], width=camera_dict['width'],
                                    height=camera_dict['height'], params=camera_dict['params']))
            camera_idxs.append(idx)
            cam_from_rig = np.eye(4)
            cam_from_rig[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(qvec_idx))
            cam_from_rig[:3, 3] = tvec_idx
            cams_from_rig.append(pycolmap.Rigid3d(cam_from_rig[:3, :]))
        pycolmap_estimation_options = dict(ransac=dict(max_error=max_error, min_inlier_ratio=min_inlier_ratio,
                                           min_num_trials=min_num_iterations, max_num_trials=max_num_iterations,
                                           confidence=confidence))
        ret = pycolmap.absolute_pose_estimation(points2D, points3D, pycolmap_cameras, camera_idxs, cams_from_rig,
                                                estimation_options=pycolmap_estimation_options)
        if ret is None:
            ret = {'success': False}
        else:
            ret['success'] = True
            if callable(ret['cam_from_world'].matrix):
                retmat = ret['cam_from_world'].matrix()
            else:
                retmat = ret['cam_from_world'].matrix
            ret['qvec'] = quaternion.from_rotation_matrix(retmat[:3, :3])
            ret['tvec'] = retmat[:3, 3]
        return ret


def pose_refinement(tvec, qvec, points2D, points3D, inlier_mask, camera_dict):
    assert has_pycolmap
    if compatibility_mode:
        return pycolmap.pose_refinement(tvec, qvec, points2D, points3D, inlier_mask, camera_dict)
    else:
        # pycolmap.pose_refinement(cam_from_world, points2D, points3D, inlier_mask, camera)
        pycolmap_camera = pycolmap.Camera(model=camera_dict['model'],
                                          width=camera_dict['width'], height=camera_dict['height'],
                                          params=camera_dict['params'])
        cam_from_world = np.eye(4)
        cam_from_world[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(qvec))
        cam_from_world[:3, 3] = tvec
        cam_from_world_pycolmap = pycolmap.Rigid3d(cam_from_world[:3, :])
        ret = pycolmap.pose_refinement(cam_from_world_pycolmap, points2D, points3D, inlier_mask, pycolmap_camera)
        if ret is None:
            ret = {'success': False}
        else:
            ret['success'] = True
            if callable(ret['cam_from_world'].matrix):
                retmat = ret['cam_from_world'].matrix()
            else:
                retmat = ret['cam_from_world'].matrix
            ret['qvec'] = quaternion.from_rotation_matrix(retmat[:3, :3])
            ret['tvec'] = retmat[:3, 3]
        return ret
