import os
import json
import re

import pandas as pd
import numpy as np

from lastfm.data_processing import sliding_window
from lastfm.baseline import AverageModel
from lastfm.helpers import timer
from lastfm.redis_connect import r, store_user_embedding_to_redis

curr_path = os.path.dirname(os.path.abspath(__file__))


def store_embedding(embedding, model_name, data_id=0):
    if not os.path.isdir(f"{curr_path}/embeddings/{model_name}"):
        os.mkdir(f"{curr_path}/embeddings/{model_name}")
    with open(
        f"{curr_path}/embeddings/{model_name}/embedding_{data_id}.json", "w"
    ) as f:
        json.dump(embedding, f)


def track_embedding(data_generator, model):
    for data, data_id in data_generator:
        embedding = model(data)
        store_embedding(embedding, model.__str__(), data_id)
        store_user_embedding_to_redis(embedding, model.__str__(), data_id)


def load_user_embedding_json(model_name):
    dir_path = f"{curr_path}/embeddings/{model_name}/"
    n_windows = len(os.listdir(dir_path))

    for i in range(n_windows):
        fname = dir_path + f"embedding_{i+1}.json"
        with open(fname) as f:
            embeddings = json.load(f)
        yield i, embeddings


@timer
def track_user_redis(uid, model_name):
    def find_window_id(x):
        x = x.decode("utf-8")
        res = re.search(f"{model_name}:window_(.+?):uid_{uid}", x).group(1)
        return int(res)

    with r.pipeline() as pipe:
        keys = sorted(
            list(r.scan_iter(f"{model_name}:*:uid_{uid}")),
            key=lambda x: find_window_id(x),
        )
        user_embeddings = np.zeros((len(keys), 32))
        print(user_embeddings.shape)
        for i, key in enumerate(keys):
            embedding = list(map(lambda x: float(x), r.lrange(key, 0, -1)))
            user_embeddings[i] = embedding
        pipe.execute()

    active_ind = list(map(lambda x: find_window_id(x), keys))
    return user_embeddings, active_ind


@timer
def track_user_db(uid, model_name):
    from lastfm.db_connect import get_user_embedding

    user_embeddings = []
    active_ind = []
    for data_id, embedding in get_user_embedding(model_name):
        if str(uid) in embedding:
            user_embeddings.append(embedding[str(uid)])
            active_ind.append(data_id)
    return np.array(user_embeddings), active_ind


@timer
def track_user(uid, model_name):
    dir_path = f"{curr_path}/embeddings/{model_name}/"
    with open(f"{curr_path}/embeddings/{model_name}_uf_0.json") as f:
        sample_user_embedding = json.load(f)
    embedding_size = len(sample_user_embedding[list(sample_user_embedding)[0]])
    n_windows = len(os.listdir(dir_path))
    null_embedding = np.zeros(embedding_size)
    user_embeddings = np.zeros((n_windows, embedding_size))
    # n_windows = 10

    for i, embeddings in load_user_embedding_json(model_name):
        uf = embeddings.get(str(uid), null_embedding)
        user_embeddings[i] = uf
    user_embedding = np.array(user_embeddings)
    active_ind = np.where(np.all(user_embeddings, axis=1))[0]
    return user_embeddings[active_ind], active_ind


@timer
def track_user_spark(uid, model_name):
    from pyspark import SparkContext

    sc = SparkContext()
    dir_path = f"{curr_path}/embeddings/{model_name}/*"
    rdd = sc.textFile(dir_path)
    uid = 31
    embeddings = rdd.map(lambda x: json.loads(x).get(str(uid))).collect()
    return np.array(embeddings)


if __name__ == "__main__":

    # data_generator = sliding_window()
    # model = AverageModel()
    # track_embedding(data_generator, model)
    uid = 31
    # e1 = track_user(uid, "baseline")
    e2 = track_user_redis(uid, "baseline")
    e3 = track_user_db(uid, "baseline")
    print(e3)
