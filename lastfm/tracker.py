import os
import json

import pandas as pd
import numpy as np

from lastfm.data_processing import sliding_window
from lastfm.baseline import AverageModel
from lastfm.helpers import timer

curr_path = os.path.dirname(os.path.abspath(__file__))

# with open(f"{curr_path}/embeddings/artist_embeddings.json") as f:
#     aid_embedding = json.load(f)


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
    # n_windows = len(os.listdir(dir_path))
    n_windows = 10
    user_embeddings = []
    for i in range(n_windows):
        fname = dir_path + f"embedding_{i+1}.json"
        with open(fname) as f:
            embeddings = json.load(f)
        embedding_size = len(embeddings[list(embeddings)[0]])
        null_embedding = np.zeros(embedding_size)
        uf = embeddings.get(str(uid), null_embedding)
        user_embeddings.append(uf)
    return np.array(user_embeddings)


@timer
def load_module():
    from plotly import graph_objects as go
    import plotly


if __name__ == "__main__":
    load_module()
    # data_generator = sliding_window()
    # model = AverageModel()
    # track_embedding(data_generator, model)
    uid = 31
    e = track_user(uid, "baseline")
    # fig = visualize_user_tracking(uid, e)
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # print(graphJSON)
