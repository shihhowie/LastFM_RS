import pandas as pd
import numpy as np
import json

def data_to_uad_agg():
	'''
	NOTE: Data already been processed, do not need to run this function again
	Aggregate timestamp data by user-artist-day triplets. This should greatly reduce the row counts
	from ~19,000,000 to 6,786,979 (2.5 GB to 500 mb, ~20% memory reduced)
	'''
	data = pd.read_csv("data/userid-timestamp-artid-artname-traid-traname.tsv", 
		names = ["userid", "timestamp", "artid", "artname", "traid","traname"],
		 sep='\t', header=None)

	# store mapping from artist_id to artist_name
	unique_artists = data.groupby(["artid", "artname"]).count()["traname"].reset_index()
	artist_name_ref = {row["artid"]: row["artname"] for index, row in unique_artists.iterrows()}
	with open('data/artist_ref.json', 'w') as f:
		json.dump(artist_name_ref, f)

	# aggregate
	data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.floor('d')
	uad_agg_data = data.groupby(["userid", "artid", "timestamp"]).count()["traname"].reset_index()
	uad_agg_data.columns = ["user_id", "artist_id", "date", "listens"]
	uad_agg_data.to_csv("data/user_id-artist_id-date-listens.tsv", sep = '\t')

def data_to_index():
	'''
	NOTE: already processed, do not need to run again
	Convert user_id, artist_id to index, for embedding in tensorflow
	Store user and artist dictionary and processed data	
	'''
	data = pd.read_csv("data/user_id-artist_id-date-listens.tsv", sep = '\t', header=0)
	unique_users = np.unique(data["user_id"])
	unique_artists = np.unique(data["artist_id"])
	user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}
	artist_mapping = {artist_id: i for i, artist_id in enumerate(unique_artists)}
	data["user_index"] = [user_mapping[user_id] for user_id in data["user_id"]]
	data["artist_index"] = [artist_mapping[artist_id] for artist_id in data["artist_id"]]
	indexed_data = data[["user_index","artist_index","date","listens"]]

	#Write dicts to json files
	with open('data/user_dict.json', 'w') as f:
		json.dump(user_mapping, f)
	with open('data/artist_dict.json', 'w') as f:
		json.dump(artist_mapping, f)
	#Save indexed data in csv
	indexed_data.to_csv("data/daily_listens.tsv", sep = '\t')

def get_data():
	data = pd.read_csv("data/daily_listens.tsv", sep = '\t', header=0)
	return(data)

def get_dicts():
	with open('data/user_dict.json') as f:
		user_dict = json.load(f)
	with open('data/artist_dict.json') as f:
		artist_dict = json.load(f)
	return(user_dict, artist_dict)

def get_artist_refs():
	with open('data/artist_ref.json') as f:
		artist_ref = json.load(f)
	return(artist_ref)
	
data_to_index()
