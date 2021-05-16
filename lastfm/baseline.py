import json
import os

import numpy as np
import pandas as pd

from lastfm.data_processing import sliding_window
from lastfm.data import get_processed_data

curr_path = os.path.dirname(os.path.abspath(__file__))
with open(f"{curr_path}/mappings/aid_to_artistid.json") as f:
    aid_to_artistid = json.load(f)
with open(f"{curr_path}/mappings/artistid_to_aname.json") as f:
    artist_to_aname = json.load(f)
with open(f"{curr_path}/mappings/uid_to_userid.json") as f:
    uid_to_userid = json.load(f)
with open(f"{curr_path}/embeddings/artist_embeddings.json") as f:
    aid_embedding = json.load(f)


def process_encoding(data):
    """
    Process window data to be fed into model
        - Aggregate data by grouping by user and artists
        - scaling count so that each it represents proportion of total listens
            this can be interpretted as the probability that user will listen to artist
    """

    agg_data = data.groupby(["uid", "aid"])["count"].sum().rename("count").reset_index()
    total_listens = agg_data.groupby("uid")["count"].transform(np.sum)
    agg_data["target"] = agg_data["count"] / total_listens
    agg_data.drop(
        ["count"],
        axis=1,
        inplace=True,
    )
    # print(agg_data)
    return agg_data


class average_model:
    """
    Baseline model that takes the aggregated window, and for every user, take the
    weighted average of the a2v embedding vector
    """

    def __init__(self, embedding_size=32, aid_embedding=aid_embedding):
        self.embedding_size = embedding_size
        self.aid_embedding = aid_embedding

    def __call__(self, data):
        data = process_encoding(data)
        n_aid = len(self.aid_embedding)
        aid_embedding_matrix = np.zeros((n_aid, self.embedding_size))
        for i in range(n_aid):
            aid_embedding_matrix[i] = self.aid_embedding[str(i)]

        uid_embedding = {}
        for uid, history in data.groupby("uid"):
            user_vector = np.zeros(aid_embedding_matrix.shape[1])
            for i, row in history.iterrows():
                user_vector += aid_embedding_matrix[int(row["aid"])] * row["target"]
            uid_embedding[str(uid)] = user_vector.tolist()
        return uid_embedding

    def __str__(self):
        return "baseline"


def store_embedding(embedding, data_id=0):
    with open(f"{curr_path}/embeddings/baseline_uf_{data_id}.json", "w") as f:
        json.dump(embedding, f)


if __name__ == "__main__":
    data = pd.read_csv("../data/test_window.csv")
    res = average_model(data)
    store_embedding(res)
