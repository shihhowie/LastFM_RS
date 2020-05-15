import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Add, concatenate, Dot, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.losses import MSE
from data_processor import get_processed_data, feed_data
from sklearn import preprocessing
import yaml
import sys

'''
TODO:
Extracting latent features and correct for transformation
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
		embeddings_regularizer = tf.keras.regularizers.l2(l=0.01), name="user_embedding")(user)
	user_bias = Embedding(n_users, 1)(user)
	artist_embedding = Embedding(n_artists, n_latent,
		embeddings_regularizer = tf.keras.regularizers.l2(l=0.01),  name="artist_embedding")(artist)
	artist_bias = Embedding(n_artists, 1)(artist)

	mf_vector = multiply([user_embedding, artist_embedding])
	logits = Dense(1)(mf_vector)
	output = Add(name="predicted_preference")([logits, user_bias, artist_bias])
	model = Model([user, artist], output)
	model.summary()
	sys.stdout.flush()
	return(model)

def PCA(X, standardize = False):
	'''normalize?
	X has shape n_users x n_latent features
	u should has same shape as X, since n_latent < n_users '''
	if standardize:
		X = preprocessing.scale(X)
	u, s, v = np.linalg.svd(X, full_matrices=False)
	return(u,s,v)

class CFModel:
	def __init__(self, params):
		self.model = cf_model(params)
		self.params = params

	def train(self, data):
		x_train, y_train = data[0]
		x_test, y_test = data[1]
		self.model.compile(optimizer = 'adam', loss = MSE)
		history = self.model.fit(x_train, y_train, 
			batch_size=self.params["batch_size"], epochs=self.params["epochs"], 
			validation_data = (x_test, y_test))
		self.history = history.history

	def get_artist_features(self):
		'''Extract artist embedding from model and transform to PCA'''
		artists = np.arange(self.params["n_artists"])
		artist_laten_matrix = self.model.get_layer("artist_embedding")(artists).numpy()
		#PCA - return u and s, or just u?
		artist_pca = PCA(artist_laten_matrix)
		return(artist_pca[0])

	def get_user_features(self):
		'''Extract user embedding from model and transform to PCA'''
		users = np.arange(self.params["n_users"])
		user_latent_matrix = self.model.get_layer("user_embedding")(users).numpy()
		#PCA 
		user_pca = PCA(user_latent_matrix)
		return(user_pca[0])



if __name__ == "__main__":
	data = pd.read_csv("data/data_example.tsv", sep="\t", header=0)
	#must include all users and artists in embedding even if they are not present in a particular month
	n_users = max(data["user_index"])+1
	n_artists = max(data["artist_index"])+1
	data = data.loc[data["date"].isin(np.arange(10,15))]
	data_slices = feed_data(data, 1)

	with open("config/model.yaml") as model_config:
		model_info = yaml.load(model_config, Loader = yaml.SafeLoader)
	params = model_info["default"]
	params["n_users"] = n_users
	params["n_artists"] = n_artists
	print(params)
	model = CFModel(params)
	users_latent_records = pd.DataFrame()
	for i, data_slice in enumerate(data_slices):
		model.train(data_slice)
		users_latent_factors = pd.DataFrame(model.get_user_features(), 
			columns = ["pca_"+str(i) for i in range(params["n_latent"])])
		users_latent_records = pd.concat([users_latent_records, users_latent_factors], axis=0)
	users_latent_records.to_csv("results/user_latent_record_test.csv")


