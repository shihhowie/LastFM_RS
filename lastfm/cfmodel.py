import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from data_processing import sliding_window
from data import get_processed_data

curr_path = os.path.dirname(os.path.abspath(__file__))
with open(f"{curr_path}/mappings/aid_to_artistid.json") as f:
    aid_to_artistid = json.load(f)
with open(f"{curr_path}/mappings/artistid_to_aname.json") as f:
    artist_to_aname = json.load(f)
with open(f"{curr_path}/mappings/uid_to_userid.json") as f:
    uid_to_userid = json.load(f)


def add_negative_samples(data, neg_size=5):
    """
    insert negative samples into data
    for users, only sample users in the same window
    for artists, sample from all artists
    """
    # neg_size = number of negative examples per user? or per pair
    unique_uids = pd.unique(data["uid"])
    n_aid = max(aid_to_artistid)
    neg_uid = np.random.choice(unique_uids, size=len(data) * neg_size)
    neg_aid = np.random.randint(0, high=n_aid, size=len(data) * neg_size)
    neg_df = pd.DataFrame({"uid": neg_uid, "aid": neg_aid})
    neg_df["target"] = 0

    full_df = pd.concat([data, neg_df], axis=0)
    full_df.drop_duplicates(subset=["uid", "aid"], keep="first", inplace=True)
    # print(full_df)
    return full_df


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
    agg_data = add_negative_samples(agg_data)
    return agg_data


class purchaseEmbedding(keras.Model):
    def __init__(self, n_uid, n_aid, embedding_size=32, **kwargs):
        super().__init__(**kwargs)
        self.n_uid = n_uid
        self.n_aid = n_aid
        self.embedding_size = embedding_size
        self.uid_embedding = layers.Embedding(
            n_uid,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.uid_bias = layers.Embedding(n_uid, 1)
        self.aid_embedding = layers.Embedding(
            n_aid,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.aid_bias = layers.Embedding(n_aid, 1)

    def call(self, inputs):
        uid_vector = self.uid_embedding(inputs[:, 0])
        uid_bias = self.uid_bias(inputs[:, 0])
        aid_vector = self.aid_embedding(inputs[:, 1])
        aid_bias = self.aid_bias(inputs[:, 1])
        dot_product = tf.tensordot(uid_vector, aid_vector, 2)
        # Add all the components (including bias)
        x = dot_product + uid_bias + aid_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


def prepare_training(data, n_uid, n_aid):

    np.random.seed(0)
    mask = np.random.rand(len(data)) < 0.8
    train, test = data.loc[mask], data.loc[~mask]
    x_train, y_train = train[["uid", "aid"]].values, train["target"].values
    x_test, y_test = test[["uid", "aid"]].values, test["target"].values

    model = purchaseEmbedding(n_uid, n_aid, embedding_size=32)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=0.001),
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=8,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    return history, model


def plot_training(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def extract_features(model, uid_to_userid, aid_to_artistid):
    uid_features = model.layers[0].get_weights()[0]
    aid_features = model.layers[2].get_weights()[0]
    userid_feature_map = {uid: uid_features[int(uid)].tolist() for uid in uid_to_userid}
    with open("../data/user_features.json", "w") as f:
        json.dump(userid_feature_map, f)

    artistid_feature_map = {
        aid: aid_features[int(aid)].tolist() for aid in aid_to_artistid
    }
    with open("../data/artist_features.json", "w") as f:
        json.dump(artistid_feature_map, f)
    return uid_features, aid_features


if __name__ == "__main__":
    data = pd.read_csv("../data/test_window.csv")
    data = process_encoding(data)
    history, model = prepare_training(data, len(uid_to_userid), len(aid_to_artistid))
    extract_features(model, uid_to_userid, aid_to_artistid)
