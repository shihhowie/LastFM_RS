import os
import json

import pandas as pd
import numpy as np

from data_processing import sliding_window
from baseline import average_model

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


if __name__ == "__main__":
    data_generator = sliding_window()
    model = average_model()
    track_embedding(data_generator, model)
