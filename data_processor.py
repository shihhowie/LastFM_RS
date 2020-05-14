import numpy as np
import pandas as pd
from data import *
import yaml
'''
TODO:
1. Add sliding window filter
'''

def get_process_data(process_name = "default_process"):
	'''
	load data from data.py and process it according to configuration in config/data.yaml
	process name is the name of the configuration in config/data.yaml'''
	data = get_data()
	with open("config/data.yaml") as data_config:
		data_info = yaml.load(data_config, Loader = yaml.SafeLoader)
	config = data_info[process_name]
	data = filter_artist(data, config["artist_filter"])
	data = aggregate_counts(data, config["agg_level"])
	data = normalize_counts(data)
	data = add_negative_samples(data, config["n_neg_samples"])
	return(data)

def filter_artist(data, n_users=5):
	'''
	Filter out artists who are no listened to by more than n_users
	Only 1000 users in experiments, we only have to filter artists
	'''
	users_per_artist = data.groupby("artist_index").count()["user_index"].reset_index()
	filtered_artists = users_per_artist["artist_index"].loc[users_per_artist["user_index"]>n_users]
	filtered_data = data[data["artist_index"].isin(filtered_artists)]
	return(filtered_data)

def aggregate_counts(data, level):
	'''
	Aggregate listening data to week level or month level
	level should be a string
	'''
	data["date"] = pd.to_datetime(data["date"])
	base_year = min(data["date"].dt.year)
	if level == "week": 
		data["date"] = 52*(data["date"].dt.year-base_year)+data["date"].dt.weekofyear
		agg_data = data.groupby(["user_index", "artist_index", "date"]).sum()["listens"].reset_index()
	elif level == "month":
		data["date"] = 12*(data["date"].dt.year-base_year)+data["date"].dt.month
		agg_data = data.groupby(["user_index", "artist_index", "date"]).sum()["listens"].reset_index()
	return(agg_data)

def normalize_counts(data):
	'''normalize weekly/monthly aggregated data via tf-idf'''

	#tf part - total listens for each user in a given date(week/month)
	total_listens = data.groupby(["user_index", "date"])["listens"].transform(np.sum)
	listen_frequency = data["listens"]/total_listens
	#idf part - total listeners for an artist adjusted by total number of listeners
	total_listeners = data.groupby(["artist_index", "date"])["user_index"].transform('count')
	all_listeners = data.groupby("date")["user_index"].transform('count')
	inverse_listener_frequency = np.log(all_listeners/total_listeners)
	data["value"] = listen_frequency*inverse_listener_frequency
	return(data[["user_index","artist_index","date","value"]])

def add_negative_samples(data, n_negs):
	'''
	add negative_samples to dataset
	usually a collaborative filtering model treats all non-listened artists as negative samples,
	but that is too data intensive, instead we just randomly sample some artists that a user has not
	listened to as negative samples.
	for every positive sample, we have n_negs negative samples
	'''
	n_artists = max(data["artist_index"])+1
	n_users = max(data["user_index"])+1
	n_dates = max(data["date"])+1
	n_entries = len(data)
	users_neg = np.random.randint(n_users, size=n_entries*n_negs)
	artists_neg = np.random.randint(n_artists, size=n_entries*n_negs)
	dates_neg = np.random.randint(n_dates, size=n_entries*n_negs)
	values_neg = np.zeros(n_entries*n_negs)
	data_neg = pd.DataFrame({"user_index": users_neg,
		"artist_index": artists_neg,
		"date": dates_neg,
		"value": values_neg})
	data_aug = pd.concat([data, data_neg], axis=0)
	#remove duplicates in negative examples
	data_aug = data_aug.groupby(["user_index", "artist_index", "date"]).max()["value"].reset_index()
	return(data_aug)

