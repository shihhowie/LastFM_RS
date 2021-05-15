import json
import pandas as pd
import numpy as np


def read_raw_data():
    data = pd.read_csv(
        "../data/userid-timestamp-artid-artname-traid-traname.tsv",
        names=[
            "userid",
            "timestamp",
            "artistid",
            "artistname",
            "trackid",
            "trackname",
        ],
        sep="\t",
        header=None,
    )
    return data


def process_raw_data():
    """
    Remove irrelavant columns for analysis
        - trackid
        - trackname
    store artistid to artistname dictionary
    convert userid and artistid to uid and aid
    Convert timestamp to date
    Aggregate grouping by userid, trackid, artistid
    """
    data = read_raw_data()

    aid_name_pairs = data.groupby(["artistid", "artistname"]).indices.keys()
    aid_name_dict = {aid: aname for aid, aname in aid_name_pairs}

    with open("mappings/artistid_to_aname.json", "w") as f:
        json.dump(aid_name_dict, f)

    artistids = pd.unique(data["artistid"])
    aid_to_artistid = {aid: artistid for aid, artistid in enumerate(artistids)}
    artistid_to_aid = {artistid: aid for aid, artistid in enumerate(artistids)}

    with open("mappings/aid_to_artistid.json", "w") as f:
        json.dump(aid_to_artistid, f)

    userids = pd.unique(data["userid"])
    uid_to_userid = {uid: userid for uid, userid in enumerate(userids)}
    userid_to_uid = {userid: uid for uid, userid in enumerate(userids)}

    with open("mappings/uid_to_userid.json", "w") as f:
        json.dump(uid_to_userid, f)

    data["uid"] = data["userid"].map(userid_to_uid)
    data["aid"] = data["artistid"].map(artistid_to_aid)

    data.drop(
        ["userid", "artistid", "trackid", "trackname", "artistname"],
        axis=1,
        inplace=True,
    )
    data.to_csv("../data/raw_data.tsv", sep="\t", index=False)
    return data


def agg_data(data):
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce").dt.date
    agg_data = (
        data[["uid", "aid", "timestamp"]].value_counts().rename("count").reset_index()
    )
    agg_data.to_csv("../data/processed_data.tsv", sep="\t", index=False)
    return agg_data


def get_processed_data():
    data = pd.read_csv("../data/processed_data.tsv", sep="\t")
    return data


def get_data():
    data = pd.read_csv("../data/raw_data.tsv", sep="\t")
    return data


def get_data_with_spark():
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import to_timestamp

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.options(header=True, delimiter="\t").csv("../data/raw_data.tsv")
    df.select(to_timestamp(df.timestamp, "yyyy-MM-dd HH:mm:ss")).collect()
    return df


if __name__ == "__main__":
    print(get_data_with_spark().show())
