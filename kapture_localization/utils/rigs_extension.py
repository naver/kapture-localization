# Copyright 2021-present NAVER Corp. Under BSD 3-clause license
from typing import List, Dict
import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
from kapture_localization.utils.logging import getLogger


def get_top_level_rig_ids(rigs: kapture.Rigs) -> List[str]:
    """
    get all top level rig ids (top level == only in the left column of rigs.txt)
    """
    rig_low_level = set().union(*[list(k.keys()) for k in rigs.values()])
    return [rigid for rigid in rigs.keys() if rigid not in rig_low_level]


def get_all_cameras_from_rig_ids(rig_ids: List[str],
                                 sensors: kapture.Sensors,
                                 rigs: kapture.Rigs) -> Dict[str, Dict[str, kapture.PoseTransform]]:
    """
    get a dict {rig_id: {camera_id: relative pose}} -> flattened version of rigs
    """
    # recursively get all camera ids for the rig ids
    camera_list = {}
    max_depth = 10
    for rig_id in rig_ids:
        camera_list[rig_id] = {}
        subrig_ids = [(rig_id, kapture.PoseTransform())]
        for _ in range(max_depth + 1):
            if len(subrig_ids) == 0:
                break
            subrig_ids_next = []
            for rig_id_l1, relative_transform_l1 in subrig_ids:
                if rig_id_l1 in sensors and \
                        isinstance(sensors[rig_id_l1], kapture.Camera):
                    camera_list[rig_id][rig_id_l1] = relative_transform_l1
                if rig_id_l1 in rigs:
                    for rig_id_l2, relative_transform_l2 in rigs[rig_id_l1].items():
                        # relative_transform_l2 -> id_l1 to id_l2
                        # relative_transform_l1 -> rig to id_l1
                        # relative_transform_l1_l2 -> rig to id_l2
                        relative_transform_l1_l2 = kapture.PoseTransform.compose(
                            [relative_transform_l2, relative_transform_l1])
                        subrig_ids_next.append((rig_id_l2, relative_transform_l1_l2))
            subrig_ids = subrig_ids_next

    # cleanup empty rigs
    final_camera_list = {}
    for rig_id in camera_list.keys():
        if len(camera_list[rig_id]) > 0:
            final_camera_list[rig_id] = camera_list[rig_id]
        else:
            getLogger().debug(f'{rig_id} removed from list because there was not any camera')
    return final_camera_list
