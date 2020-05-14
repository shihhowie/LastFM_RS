import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Add, concatenate, Dot, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.losses import MSE
from data_processor import get_processed_data
import sys

def cf_model(params):
	if params.get("use_seed"):
		tf.set_seed(10245)
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

if __name__ == "__main__":
	data = get_processed_data()
	n_users = max(data["user_index"])+1
	n_artists = max(data["artist_index"])+1
	data = data.loc[data["date"]==10]
	train_ind = np.random.rand(len(data)) > 0.1
	train_data = data.loc[train_ind]
	test_data = data.loc[~train_ind]
	x_train = [train_data["user_index"].values, train_data["artist_index"].values]
	x_test = [test_data["user_index"].values, test_data["artist_index"].values]
	y_train = train_data["value"].values
	y_test = test_data["value"].values

	print(len(data))
	params = {}
	params["user_seed"] = True
	params["n_latent"] = 10
	params["n_users"] = n_users
	params["n_artists"] = n_artists
	print(params)
	model = cf_model(params)
	model.compile(optimizer = 'adam', loss = MSE)
	history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data = (x_test, y_test))
	print(history.history)

