import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture
from kapture_localization.utils.logging import getLogger
from kapture.io.csv import table_from_file
from collections import OrderedDict
from typing import Dict, List, Tuple


def get_pairs_from_file(pairsfile_path: str,
                        query_records: kapture.RecordsCamera = None,
                        map_records: kapture.RecordsCamera = None,) -> List[Tuple[str, str]]:
    """
    read a pairs file (csv with 3 fields, name1, name2, score) and return the list of matches

    :param pairsfile_path: path to pairsfile
    :type pairsfile_path: str
    """
    getLogger().info('reading pairs from pairsfile')
    if query_records is not None:
        query_images = set(query_records.data_list())
    else:
        query_images = None
    if map_records is not None:
        map_images = set(map_records.data_list())
    else:
        map_images = None

    image_pairs = []
    with open(pairsfile_path, 'r') as fid:
        table = table_from_file(fid)
        for query_name, map_name, _ in table:  # last field score is not used
            if query_images is not None and query_name not in query_images:
                continue
            if map_images is not None and map_name not in map_images:
                continue
            if query_name != map_name:
                image_pairs.append((query_name, map_name) if query_name < map_name else (map_name, query_name))
    # remove duplicates without breaking order
    image_pairs = list(OrderedDict.fromkeys(image_pairs))
    return image_pairs


def get_ordered_pairs_from_file(pairsfile_path: str,
                                query_records: kapture.RecordsCamera = None,
                                map_records: kapture.RecordsCamera = None,
                                topk_override=None) -> Dict[str, List[Tuple[str, float]]]:
    """
    read pairfile and return a list of pairs (keep duplicates, order is query, map)
    """
    getLogger().info('reading pairs from pairsfile')
    if query_records is not None:
        query_images = set(query_records.data_list())
    else:
        query_images = None
    if map_records is not None:
        map_images = set(map_records.data_list())
    else:
        map_images = None

    image_pairs = {}
    with open(pairsfile_path, 'r') as fid:
        table = table_from_file(fid)
        for query_name, map_name, score in table:
            if query_images is not None and query_name not in query_images:
                continue
            if map_images is not None and map_name not in map_images:
                continue
            if query_name not in image_pairs:
                image_pairs[query_name] = []
            image_pairs[query_name].append((map_name, float(score)))
    for k in image_pairs.keys():
        sorted_by_score = list(sorted(image_pairs[k], key=lambda x: x[1], reverse=True))
        if topk_override is not None and topk_override > len(sorted_by_score):
            getLogger().debug(f'image {k} has {len(sorted_by_score)} pairs, less than topk={topk_override}')
        elif topk_override is not None:
            sorted_by_score = sorted_by_score[:topk_override]
        image_pairs[k] = sorted_by_score
    return image_pairs
