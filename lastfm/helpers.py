import os
import json

import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
with open(f"{curr_path}/mappings/uid_to_userid.json") as f:
    uid_to_userid = json.load(f)


def convert_to_matrix(embedding_map):
    n_uids = len(uid_to_userid)
    embedding_size = len(list(embedding_map.values())[0])
    embedding_matrix = np.zeros((n_uids, embedding_size))
    for uid, embedding_vector in embedding_map.items():
        embedding_matrix[int(uid)] = embedding_vector
    return embedding_matrix
