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

    with open("artistid_to_aname.json", "w") as f:
        json.dump(aid_name_dict, f)

    artistids = pd.unique(data["artistid"])
    aid_to_artistid = {aid: artistid for aid, artistid in enumerate(artistids)}
    artistid_to_aid = {artistid: aid for aid, artistid in enumerate(artistids)}

    with open("aid_to_artistid.json", "w") as f:
        json.dump(aid_to_artistid, f)

    userids = pd.unique(data["userid"])
    uid_to_userid = {uid: userid for uid, userid in enumerate(userids)}
    userid_to_uid = {userid: uid for uid, userid in enumerate(userids)}

    with open("uid_to_userid.json", "w") as f:
        json.dump(uid_to_userid, f)

    data["uid"] = [userid_to_uid[x] for x in data["userid"]]
    data["aid"] = [artistid_to_aid[x] for x in data["artistid"]]

    data.drop(
        ["userid", "artistid", "trackid", "trackname", "artistname"],
        axis=1,
        inplace=True,
    )

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce").dt.date
    agg_data = (
        data[["uid", "aid", "timestamp"]].value_counts().rename("count").reset_index()
    )
    agg_data.to_csv("../data/processed_data.tsv", sep="\t", index=False)


def get_processed_data():
    data = pd.read_csv("../data/processed_data.tsv", sep="\t")
    return data


if __name__ == "__main__":
    print(get_processed_data())
