# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
from typing import List, Optional, Dict, Tuple
import numpy as np

from kapture_localization.localization.DuplicateCorrespondencesStrategy import DuplicateCorrespondencesStrategy
from kapture_localization.localization.RerankCorrespondencesStrategy import RerankCorrespondencesStrategy

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
from kapture.io.features import get_matches_fullpath, image_matches_from_file
from kapture.io.tar import TarCollection


def get_correspondences(kapture_data: kapture.Kapture, keypoints_type: str,
                        kapture_path: str, tar_handlers: TarCollection,
                        img_query: str, pairs: List[str],
                        point_id_from_obs: Dict[Tuple[str, int], int],
                        kpts_query: Optional[np.ndarray],
                        kpts_query_undistorted: Optional[np.ndarray],
                        duplicate_strategy: DuplicateCorrespondencesStrategy,
                        rerank_strategy: RerankCorrespondencesStrategy):
    """
    get 2D-3D correspondences for a given query image, a list of paired map images, and a kapture map
    """
    # first list all correspondences
    correspondences = {}
    for img_map in pairs:
        # get matches
        if img_query < img_map:
            assert (img_query, img_map) in kapture_data.matches[keypoints_type]
            matches_path = get_matches_fullpath((img_query, img_map), keypoints_type, kapture_path, tar_handlers)
        else:
            assert (img_map, img_query) in kapture_data.matches[keypoints_type]
            matches_path = get_matches_fullpath((img_map, img_query), keypoints_type, kapture_path, tar_handlers)
        matches = image_matches_from_file(matches_path)

        num_matches = matches.shape[0]
        corrs = []
        for m in matches:
            if img_query < img_map:
                kpid_query = m[0]
                kpid_map = m[1]
            else:
                kpid_query = m[1]
                kpid_map = m[0]
            # match_score = m[2]

            if not (img_map, kpid_map) in point_id_from_obs:
                continue
            # get 3D point
            p3did = point_id_from_obs[(img_map, kpid_map)]
            corrs.append((kpid_query, p3did))
        correspondences[img_map] = (num_matches, corrs)

    if rerank_strategy == RerankCorrespondencesStrategy.none:
        reranked_pairs = pairs
    elif rerank_strategy == RerankCorrespondencesStrategy.matches_count:
        reranked_pairs = [img_map for img_map, _ in sorted(correspondences.items(),
                                                           key=lambda x: x[1][0],
                                                           reverse=True)]
    elif rerank_strategy == RerankCorrespondencesStrategy.correspondences_count:
        reranked_pairs = [img_map for img_map, _ in sorted(correspondences.items(),
                                                           key=lambda x: len(x[1][1]),
                                                           reverse=True)]
    else:
        raise NotImplementedError(f'{rerank_strategy} not implemented')

    # N number of correspondences
    # points2D - Nx2 array with pixel coordinates
    # points3D - Nx3 array with world coordinates
    points2D = []
    points2D_undistorted = []
    points3D = []

    assigned_keypoints_ids = {}
    assigned_3d_points_ids = {}
    true_duplicates_count = 0
    same_2d_multiple_3d_count = 0
    same_2d_multiple_3d_max = 0
    same_3d_multiple_2d_count = 0
    same_3d_multiple_2d_max = 0
    rejected_correspondences = 0
    for img_map in reranked_pairs:
        for kpid_query, p3did in correspondences[img_map][1]:
            if kpid_query in assigned_keypoints_ids and p3did in assigned_keypoints_ids[kpid_query]:
                true_duplicates_count += 1
                if duplicate_strategy == DuplicateCorrespondencesStrategy.ignore or \
                        duplicate_strategy == DuplicateCorrespondencesStrategy.ignore_strict or \
                        duplicate_strategy == DuplicateCorrespondencesStrategy.ignore_same_kpid or \
                        duplicate_strategy == DuplicateCorrespondencesStrategy.ignore_same_p3did:
                    rejected_correspondences += 1
                    continue
            elif duplicate_strategy == DuplicateCorrespondencesStrategy.ignore and \
                    (kpid_query in assigned_keypoints_ids or p3did in assigned_3d_points_ids):
                rejected_correspondences += 1
                continue
            else:
                if duplicate_strategy == DuplicateCorrespondencesStrategy.ignore_same_kpid and \
                        kpid_query in assigned_keypoints_ids:
                    rejected_correspondences += 1
                    continue
                elif kpid_query not in assigned_keypoints_ids:
                    assigned_keypoints_ids[kpid_query] = {p3did}
                else:
                    # p3did not in assigned_keypoints_ids[kpid_query]
                    same_2d_multiple_3d_count += 1
                    assigned_keypoints_ids[kpid_query].add(p3did)
                    same_2d_multiple_3d_max = max(same_2d_multiple_3d_max, len(assigned_keypoints_ids[kpid_query]))

                if duplicate_strategy == DuplicateCorrespondencesStrategy.ignore_same_p3did and \
                        p3did in assigned_3d_points_ids:
                    rejected_correspondences += 1
                    continue
                elif p3did not in assigned_3d_points_ids:
                    assigned_3d_points_ids[p3did] = {kpid_query}
                else:
                    # kpid_query not in assigned_3d_points_ids[p3did]
                    same_3d_multiple_2d_count += 1
                    assigned_3d_points_ids[p3did].add(p3did)
                    same_3d_multiple_2d_max = max(same_3d_multiple_2d_max, len(assigned_3d_points_ids[p3did]))

            if kpts_query is not None:
                kp_query = kpts_query[int(kpid_query)]
                points2D.append(kp_query[0:2])
            if kpts_query_undistorted is not None:
                kp_query_undistorted = kpts_query_undistorted[int(kpid_query)]
                points2D_undistorted.append(kp_query_undistorted[0:2])
            p3d_map = kapture_data.points3d[p3did]
            points3D.append(p3d_map[0:3])

    stats = {
        "true_duplicates_count": true_duplicates_count,
        "same_2d_multiple_3d_count": same_2d_multiple_3d_count,
        "same_2d_multiple_3d_max": same_2d_multiple_3d_max,
        "same_3d_multiple_2d_count": same_3d_multiple_2d_count,
        "same_3d_multiple_2d_max": same_3d_multiple_2d_max,
        "rejected_correspondences": rejected_correspondences
    }
    return points2D, points2D_undistorted, points3D, stats
