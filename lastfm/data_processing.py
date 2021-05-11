import json
import datetime
import pandas as pd
import numpy as np

from data import get_processed_data, get_data


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
    while end_window < max_date:
        window = data.loc[
            (data["timestamp"] >= beg_window.strftime("%Y-%m-%d"))
            & (data["timestamp"] < end_window.strftime("%Y-%m-%d"))
        ]
        yield window
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


def find_sessions(history, filter=500):
    history = history[::-1]
    history["diff"] = history["timestamp"].diff().dt.seconds.fillna(0)
    session = []
    for i, row in history.iterrows():
        if row["diff"] < filter:
            session.append(row["aid"])
        else:
            session = remove_dup(session)
            session = list(map(lambda x: str(x), session))
            if len(session) > 1:
                with open("../data/session_data.txt", "a") as f:
                    f.write(" ".join(session) + "\n")
            session = []


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
    sessions = []
    for uid, history in data.groupby("uid"):
        find_sessions(history)


if __name__ == "__main__":
    print(process_sessions())
    # data = sliding_window()
    # window = next(data)
    # window.to_csv("../data/test_window.csv", index=False)
    # print(window)
    # print(next(data))
    # print(next(data))
