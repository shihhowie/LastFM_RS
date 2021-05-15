import json
import os
import shutil
import datetime

import pandas as pd
import numpy as np

from pyspark.sql.functions import pandas_udf

from lastfm.data import get_processed_data, get_data, get_data_with_spark

from lastfm.helpers import timer


@timer
def data_filtering(data, min_listens=5):
    """
    Filter data based on how many artists will be kept in the dataset
    Artists only listened to by one user will not be very useful in any model
    Also filters artists that have very few total listens
    """
    agg_data_by_artist = data.groupby("aid")["count"].sum()
    valid_artists = agg_data_by_artist.loc[agg_data_by_artist >= min_listens]
    print(
        "percent valid artists after filtering for min listens",
        len(valid_artists) / len(agg_data_by_artist),
    )
    filtered_data = data.loc[data["aid"].isin(valid_artists.index)]
    return filtered_data


def sliding_window(step=1):
    """
    This is a generator that implements sliding window procedure to generate monthly data
    Parameters
        step: how many weeks to skip over for the next window.
    """
    data = get_processed_data()
    min_date = datetime.datetime.strptime(min(data["timestamp"]), "%Y-%m-%d")
    max_date = datetime.datetime.strptime(max(data["timestamp"]), "%Y-%m-%d")

    beg_window = min_date
    end_window = min_date + datetime.timedelta(weeks=4)
    i = 1
    while end_window < max_date:
        window = data.loc[
            (data["timestamp"] >= beg_window.strftime("%Y-%m-%d"))
            & (data["timestamp"] < end_window.strftime("%Y-%m-%d"))
        ]
        yield window, i
        i += 1
        beg_window = beg_window + datetime.timedelta(weeks=step)
        end_window = end_window + datetime.timedelta(weeks=step)


def remove_dup(session):
    cleaned_session = []
    curr_artist = -1
    for artist in session:
        if artist != curr_artist:
            cleaned_session.append(artist)
            curr_artist = artist
    return cleaned_session


def write_session_data(session_data, fname=None):
    with open("../data/session_data_by_spark.txt", "a") as f:
        for session in session_data:
            if len(session) > 1:
                f.write(" ".join(session) + "\n")


def find_sessions(history, filter=500, batch_write=False):
    history = history[::-1]
    history["diff"] = history["timestamp"].diff().dt.seconds.fillna(0)
    sessions = []
    session = []
    for i, row in history.iterrows():
        if row["diff"] < filter:
            session.append(row["aid"])
        else:
            session = remove_dup(session)
            session = list(map(lambda x: str(x), session))
            if not batch_write and len(session) > 1:
                with open("../data/session_data_by_spark.txt", "a") as f:
                    f.write(" ".join(session) + "\n")
            else:
                sessions.append(session)
            session = []
    return sessions


import itertools


@timer
def process_sessions():
    """
    *opportunity to use pyspark here
    Process timeseries of user's listening record for downstream artist2vec training
        - define session
        - session represents a sentence
        - remove consecutive artists
    """
    data = get_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for uid, history in itertools.islice(data.groupby("uid"), 10):
        sessions = find_sessions(history)
        write_session_data(sessions)


def combined_spark_outputs(path):
    for fname in os.listdir(path):
        if fname.startswith("part-"):
            with open(f"{path}/{fname}") as f:
                content = f.read()
            with open(f"{path}.txt", "a") as f:
                f.write(content)
    shutil.rmtree(path)


@timer
def process_sessions_with_spark(sc, fname=None):

    # data = get_data_with_spark()
    data = get_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    uid_groups = itertools.islice(data.groupby("uid"), 10)
    rdd = sc.parallelize(uid_groups)
    curr_path = os.path.dirname(os.path.abspath(__file__))

    # sessions = (
    #     rdd.flatMap(lambda x: find_sessions(x[1], batch_write=True))
    #     .filter(lambda x: len(x) > 1)
    #     .collect()
    # )
    # write_session_data(sessions)

    rdd.flatMap(lambda x: find_sessions(x[1], batch_write=True)).filter(
        lambda x: len(x) > 1
    ).saveAsTextFile(fname)
    combined_spark_outputs(fname)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(__file__))

    from pyspark import SparkContext

    sc = SparkContext()
    fname = f"{curr_path}/../data/session_data_by_spark2"
    sessions = process_sessions_with_spark(sc, fname)
    # combined_spark_outputs(fname)
    # print(sessions)
    # process_sessions()
    # data = sliding_window()
    # window = next(data)
    # window.to_csv("../data/test_window.csv", index=False)
    # print(window)
    # print(next(data))
    # print(next(data))
