import numpy as np
import pandas as pd
from data import *
import yaml
'''
Daily data -> weekly/monthly sum aggregate -> filter out artists ->  add negative sample
-> sliding window -> normalize by window  -> feed window/whole data with re-indexing
if not fed in slices -> need to reset user id
'''

def aggregate_counts(data, level):
	'''
	Aggregate listening data to week level or month level
	level should be a string
	'''
	data["date"] = pd.to_datetime(data["date"])
	base_year = min(data["date"].dt.year)
	if level == "week": 
		data["date"] = 52*(data["date"].dt.year-base_year)+(data["date"].dt.weekofyear-1)
		agg_data = data.groupby(["user_index", "artist_index", "date"]).sum()["listens"].reset_index()
	elif level == "month":
		data["date"] = 12*(data["date"].dt.year-base_year)+(data["date"].dt.month-1)
		agg_data = data.groupby(["user_index", "artist_index", "date"]).sum()["listens"].reset_index()
	return(agg_data)

def agg_by_week_and_month():
	'''NOTE: Already ran, do not need to run again
	aggregate data by week and month and store in tsv file'''
	data = get_data()
	data_weekly_agg = aggregate_counts(data.copy(), "week")
	data_monthly_agg = aggregate_counts(data.copy(), "month")
	data_weekly_agg.to_csv("data/weekly_agg_data.tsv", sep='\t', index=False)
	data_monthly_agg.to_csv("data/monthly_agg_data.tsv", sep='\t', index=False)

'''-------the methods above are pre-ran, do not need to re-run unless reconfigure at daily level-------'''

def filter_artist(data, n_users=5):
	'''
	Filter out artists who are no listened to by more than n_users
	Only 1000 users in experiments, we only have to filter artists
	'''
	users_per_artist = data.groupby("artist_index")["user_index"].nunique().reset_index()
	filtered_artists = users_per_artist["artist_index"].loc[users_per_artist["user_index"]>n_users]
	filtered_data = data[data["artist_index"].isin(filtered_artists)]
	return(filtered_data[["user_index","artist_index","date","listens"]])

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
	begin, end = min(data["date"]), max(data["date"])+1
	n_entries = len(data)
	users_neg = np.random.randint(n_users, size=n_entries*n_negs)
	artists_neg = np.random.randint(n_artists, size=n_entries*n_negs)
	date_neg = np.random.randint(begin, end, size=n_entries*n_negs)
	listens_neg = np.zeros(n_entries*n_negs)
	data_neg = pd.DataFrame({"user_index": users_neg,
		"artist_index": artists_neg,
		"date": date_neg,
		"listens": listens_neg})
	data_aug = pd.concat([data, data_neg], axis=0)
	#remove duplicates in negative examples
	data_aug = data_aug.groupby(["user_index", "artist_index", "date"]).max()["listens"].reset_index()
	return(data_aug)

def data_slicer(data, window = 1):
	'''slice data by date window, aggregate the metric in each window'''
	start_date,end_date = min(data["date"]), max(data["date"])+1
	n_windows = (end_date-start_date)-window
	data["window"] = 0
	sliced_data = pd.DataFrame()
	for date in range(start_date, end_date-window):
		data_slice = data.loc[data["date"].isin(np.arange(date,date+window))]
		# aggregate out repeated artits
		data_slice = data_slice.groupby(["user_index", "artist_index"])["listens"].sum().reset_index()
		data_slice["window"] = date-start_date
		sliced_data = pd.concat([sliced_data, data_slice])
	return(sliced_data)

def normalize_counts(data):
	'''normalize window aggregated data via tf-idf
		* Remove idf part for now as cf_model should account for user bias and artist bias'''
	#tf part - total listens for each user in a given window
	total_listens = data.groupby(["user_index", "window"])["listens"].transform(np.sum)
	listen_frequency = data["listens"]/total_listens
	#idf part - total listeners for an artist adjusted by total number of listeners
	# total_listeners = data.groupby(["artist_index", "date"])["user_index"].transform("count")
	# inverse_listener_frequency = np.log(992/(total_listeners+1))
	data["value"] = listen_frequency
	# remove na - since if a user-window pair is na then it is entirely negative samples
	data = data.dropna()
	return(data[["user_index","artist_index","window","value"]])

def reindexing(data):
	'''Reindex user index by coalescing with month
	Store mapping to json file'''
	user_window_id = data["user_index"].astype(str).str.cat(data["window"].astype(str),sep="#")
	user_window_map = {uw_id: i for i, uw_id in enumerate(pd.unique(user_window_id))}
	user_window_index = [user_window_map[uw_id] for uw_id in user_window_id]
	data["user_window_index"] = user_window_index
	return(data[["user_window_index", "artist_index", "value"]], user_window_map)

def feed_data(data, window = 1):
	'''
	Data feeder to model in slices by date window, split by train and test
	data slices come out in form 
	[[(train_x1, train_y1), (test_x1, test_y1)], [(train_x2, train_y2), (test_x2, test_y2)]]
	'''
	n_windows=  max(data["window"])
	data_slices = []
	for window_index in range(n_windows):
		data_slice = data[data["window"]==window_index]
		train_ind = np.random.rand(len(data_slice)) > 0.1

		train_slice = data_slice.loc[train_ind]
		train_x = [train_slice["user_index"].values, train_slice["artist_index"].values]
		train_y = train_slice["value"].values

		test_slice = data_slice.loc[~train_ind]
		test_x = [test_slice["user_index"].values, test_slice["artist_index"].values]
		test_y = test_slice["value"].values

		data_slices.append([(train_x, train_y), (test_x, test_y)])
	return(data_slices)

class DataManager:
	def __init__(self, process_name = "default_process"):
		with open("config/data.yaml") as data_config:
			data_info = yaml.load(data_config, Loader = yaml.SafeLoader)
		self.config = data_info[process_name]
		# short_cut
		self.data = pd.read_csv("data/data_example.tsv", sep="\t", header = 0)
		# self.process_data()

	def process_data(self):
		if self.config["agg_level"] == "week":
			data = pd.read_csv("data/weekly_agg_data.tsv", sep = "\t", header=0)
		elif self.config["agg_level"] == "month":
			data = pd.read_csv("data/monthly_agg_data.tsv", sep = "\t", header=0)
		data = filter_artist(data, self.config["artist_filter"])
		data = add_negative_samples(data, self.config["n_neg_samples"])
		data = data_slicer(data, self.config["window"])
		data = normalize_counts(data)
		self.data = data

	def feed_data_windows(self):
		return(feed_data(self.data))

	def get_full_data(self):
		'''get full sliced data with user_window index mapping'''
		data, user_window_dict = reindexing(self.data)
		x = [data["user_window_index"].values,data["artist_index"].values]
		y = data["value"].values
		self.user_window_dict = user_window_dict
		return(x,y)

# testing
if __name__ == "__main__":
	# agg_by_week_and_month()
	manager = data_manager()
	print(manager.get_full_data())