import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Add, concatenate, Dot, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.losses import MSE
from data_processor import DataManager
from tensorflow.keras.optimizers import Adam
import yaml
import sys
import json


'''
TODO:
Extracting latent features(DONE) and correct for transformation(Moved to other file)
Figure out good metrics for monitoring performance
Fix sparse tensor warning
Implement monitoring with tensorboard
Test out continuous training - using previous embedding as starting value
'''
def cf_model(params):
	if params.get("use_seed"):
		tf.random.set_seed(10245)
	n_users = params["n_users"]
	n_artists = params["n_artists"]
	n_latent = params["n_latent"]

	user = Input(shape=(1,))
	artist = Input(shape=(1,))

	user_embedding = Embedding(n_users, n_latent, 
		embeddings_regularizer = tf.keras.regularizers.l2(l=1e-6), name="user_embedding")(user)
	user_bias = Embedding(n_users, 1,
		embeddings_regularizer = tf.keras.regularizers.l2(l=1e-6))(user)
	artist_embedding = Embedding(n_artists, n_latent,
		embeddings_regularizer = tf.keras.regularizers.l2(l=1e-6),  name="artist_embedding")(artist)
	artist_bias = Embedding(n_artists, 1,
		embeddings_regularizer = tf.keras.regularizers.l2(l=1e-6))(artist)

	mf_vector = Dot(axes=-1)([user_embedding, artist_embedding])
	# logits = Dense(1)(mf_vector)
	output = Add(name="predicted_preference")([mf_vector, user_bias, artist_bias])
	model = Model([user, artist], output)
	model.summary()
	sys.stdout.flush()
	return(model)

class CFModel:
	def __init__(self, params):
		self.model = cf_model(params)
		# Save inital weights for re-run at different time windows
		self.model.save_weights("initial")
		self.params = params
		self.counter = 0

	def train(self, data):
		# Load initial weight every time model retrains
		self.model.load_weights("initial")
		print(data[0])
		print(data[1])
		x_train, y_train = data[0]
		x_test, y_test = data[1]
		opt = Adam(lr = self.params.get("learning_rate", 0.01))
		self.model.compile(optimizer ='adam', loss = MSE)
		history = self.model.fit(x_train, y_train, 
			batch_size=self.params["batch_size"], epochs=self.params["epochs"], 
			validation_data = (x_test, y_test))
		self.history = history.history
		self.counter += 1

	def get_latent_features(self):
		artists = np.arange(self.params["n_artists"])
		artist_laten_matrix = self.model.get_layer("artist_embedding")(artists).numpy()
		users = np.arange(self.params["n_users"])
		user_latent_matrix = self.model.get_layer("user_embedding")(users).numpy()
		return(user_latent_matrix, artist_laten_matrix)



if __name__ == "__main__":
	data_manager = DataManager()
	x, y = data_manager.get_full_data()
	with open('data/user_window_ref.json', 'w') as f:
		json.dump(data_manager.user_window_dict, f)
	print(x)
	train_ind = np.random.rand(len(y)) > 0.1
	train_x = [x[0][train_ind], x[1][train_ind]]
	train_y = y[train_ind]
	test_x = [x[0][~train_ind], x[1][~train_ind]]
	test_y = y[~train_ind]
	#must include all users and artists in embedding even if they are not present in a particular month
	n_users = max(x[0])+1
	n_artists = max(x[1])+1

	with open("config/model.yaml") as model_config:
		model_info = yaml.load(model_config, Loader = yaml.SafeLoader)
	params = model_info["default"]
	params["n_users"] = n_users
	params["n_artists"] = n_artists
	model = CFModel(params)
	model.train(data = [(train_x, train_y), (test_x, test_y)])
	users_latent_factors, artists_latent_features = model.get_latent_features()
	users_latent_factors_df = pd.DataFrame(users_latent_factors, 
		columns = ["feature_"+str(i) for i in range(params["n_latent"])])
	users_latent_factors_df.to_csv("results/user_latent_record_test2.tsv", sep='\t', index=False)

	# users_latent_records = pd.DataFrame()
	# for i, data_slice in enumerate(data_slices):
	# 	model.train(data_slice)
	# 	users_latent_factors = pd.DataFrame(model.get_user_features(), 
	# 		columns = ["feature_"+str(i) for i in range(params["n_latent"])])
	# 	# add column for window
	# 	users_latent_factors["user_index"] = np.arange(params["n_users"])
	# 	users_latent_factors["window"] = i
	# 	users_latent_records = pd.concat([users_latent_records, users_latent_factors], axis=0)
	# users_latent_records = users_latent_records
	# users_latent_records.to_csv("results/user_latent_record_test.tsv", sep='\t', index=False)


