import os
import json
import functools
import time

import numpy as np


def convert_to_matrix(embedding_map):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{curr_path}/mappings/uid_to_userid.json") as f:
        uid_to_userid = json.load(f)

    n_uids = len(uid_to_userid)
    embedding_size = len(list(embedding_map.values())[0])
    embedding_matrix = np.zeros((n_uids, embedding_size))
    for uid, embedding_vector in embedding_map.items():
        embedding_matrix[int(uid)] = embedding_vector
    return embedding_matrix


def timer(f):
    @functools.wraps(f)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        res = f(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {f.__name__!r} in {run_time:.4f} seconds")
        return res

    return wrapper_timer
