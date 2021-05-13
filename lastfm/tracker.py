import os
import json

import pandas as pd
import numpy as np
import umap
from plotly import graph_objects as go
import plotly
import matplotlib.pyplot as plt

from lastfm.data_processing import sliding_window
from lastfm.baseline import average_model

curr_path = os.path.dirname(os.path.abspath(__file__))

with open(f"{curr_path}/embeddings/artist_embeddings.json") as f:
    aid_embedding = json.load(f)


def store_embedding(embedding, name, data_id=0):
    if not os.path.isdir(f"{curr_path}/embeddings/{name}"):
        os.mkdir(f"{curr_path}/embeddings/{name}")
    with open(f"{curr_path}/embeddings/{name}/embedding_{data_id}.json", "w") as f:
        json.dump(embedding, f)


def track_embedding(data_generator, model):
    for data, data_id in data_generator:
        embedding = model(data)
        store_embedding(embedding, model.__str__(), data_id)


def track_user(uid, model_name):
    dir_path = f"{curr_path}/embeddings/{model_name}/"
    n_windows = len(os.listdir(dir_path))
    # n_windows = 10
    embedding_size = len(list(aid_embedding.values())[0])
    user_embeddings = np.zeros((n_windows, embedding_size))
    for i in range(n_windows):
        fname = dir_path + f"embedding_{i+1}.json"
        with open(fname) as f:
            embeddings = json.load(f)
        null_embedding = np.zeros(embedding_size)
        uf = embeddings.get(str(uid), null_embedding)
        user_embeddings[i] = uf
    return user_embeddings


def visualize_user_tracking(uid, user_embeddings):
    reducer = umap.UMAP()
    active_ind = np.where(np.all(user_embeddings, axis=1))[0]
    embeddings_active = user_embeddings[active_ind]
    reduced_embedding = reducer.fit_transform(embeddings_active)
    # fig, _ = plt.subplots(figsize=(12, 8))
    # plt.plot(reduced_embedding[:, 0], reduced_embedding[:, 1], "o-", alpha=0.8)
    # for i in range(len(active_ind)):
    #     plt.annotate(active_ind[i], (reduced_embedding[i, 0], reduced_embedding[i, 1]))
    # plt.title(f"user tracking for uid {uid}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reduced_embedding[:, 0],
            y=reduced_embedding[:, 1],
            text=[str(x) for x in active_ind],
            mode="lines+markers+text",
            textposition="top center",
        )
    )

    return fig


if __name__ == "__main__":
    # data_generator = sliding_window()
    # model = average_model()
    # track_embedding(data_generator, model)
    uid = 31
    e = track_user(uid, "baseline")
    fig = visualize_user_tracking(uid, e)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print(graphJSON)
