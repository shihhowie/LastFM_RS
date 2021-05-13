import os
import json

import pandas as pd
import numpy as np

from data_processing import sliding_window
from baseline import average_model

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


if __name__ == "__main__":
    # data_generator = sliding_window()
    # model = average_model()
    # track_embedding(data_generator, model)
    print(track_user(1, "baseline"))
