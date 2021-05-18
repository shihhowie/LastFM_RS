import os
import json
import functools
import time

import numpy as np

from lastfm.redis_connect import r

curr_path = os.path.dirname(os.path.abspath(__file__))


def convert_to_matrix(embedding_map):
    with open(f"{curr_path}/mappings/uid_to_userid.json") as f:
        uid_to_userid = json.load(f)

    n_uids = len(uid_to_userid)
    embedding_size = len(embedding_map[list(embedding_map)[0]])
    embedding_matrix = np.zeros((n_uids, embedding_size))
    for uid, embedding_vector in embedding_map.items():
        embedding_matrix[int(uid)] = embedding_vector
    return embedding_matrix


def get_artist_embedding(aid):

    embedding = list(map(lambda x: float(x), r.lrange(f"aid:{aid}", 0, -1)))
    return embedding


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


if __name__ == "__main__":
    # print(get_artist_embedding(31))
    pass
