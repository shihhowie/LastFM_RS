import os
import json

import pandas as pd
import numpy as np


from lastfm.data_processing import sliding_window
from lastfm.baseline import AverageModel
from lastfm.helpers import timer

curr_path = os.path.dirname(os.path.abspath(__file__))


def store_embedding(embedding, name, data_id=0):
    if not os.path.isdir(f"{curr_path}/embeddings/{name}"):
        os.mkdir(f"{curr_path}/embeddings/{name}")
    with open(f"{curr_path}/embeddings/{name}/embedding_{data_id}.json", "w") as f:
        json.dump(embedding, f)


def track_embedding(data_generator, model):
    for data, data_id in data_generator:
        embedding = model(data)
        store_embedding(embedding, model.__str__(), data_id)


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

    def get_user_embedding_matrix(fname):
        with open(fname) as f:
            embeddings = json.load(f)
        uf = embeddings.get(str(uid), null_embedding)
        return uf

    for i in range(n_windows):
        fname = dir_path + f"embedding_{i+1}.json"
        uf = get_user_embedding_matrix(fname)
        user_embeddings[i] = uf
        # user_embeddings.append(uf)
    return np.array(user_embeddings)


@timer
def track_user_spark(sc, uid, model_name):

    dir_path = f"{curr_path}/embeddings/{model_name}/*"
    rdd = sc.textFile(dir_path)
    uid = 31
    embeddings = rdd.map(lambda x: json.loads(x).get(str(uid))).collect()
    return np.array(embeddings)


if __name__ == "__main__":
    from pyspark import SparkContext

    sc = SparkContext()
    track_user_spark(sc, 31, "baseline")
    # data_generator = sliding_window()
    # model = AverageModel()
    # track_embedding(data_generator, model)
    uid = 31
    e = track_user(uid, "baseline")
    # fig = visualize_user_tracking(uid, e)
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # print(graphJSON)
