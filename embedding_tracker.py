import yaml
from data_processor import get_processed_data, feed_data
from sklearn import preprocessing
import numpy as np
import pandas as pd

'''
Track embedding features across different time window
Adjust axes by procrustes transform
'''

def PCA(X, standardize = False):
	'''normalize?
	X has shape n_users x n_latent features
	u should has same shape as X, since n_latent < n_users '''
	if standardize:
		X = preprocessing.scale(X)
	u, s, v = np.linalg.svd(X, full_matrices=False)
	return(u,s,v)

class EmbeddingTracker:
	def __init__(self):
		self.counter = 0

	def feed_data(self, data):
		'''data is in latent feature representation.'''
		if self.counter == 0:
			self.X_0 = data
		else:
			data = self.procrustes_transform(data)
		self.counter += 1
		n_rows, n_features = data.shape
		data_df = pd.DataFrame(data, columns = ["feature_"+str(i) for i in range(n_features)])
		data_df["user_index"] = np.arange(n_rows)
		data_df["window"] = self.counter
		self.data_df = data_df

	def write_to_file(self, path):
		with open("results/{}.tsv".format(path), mode='a') as f:
			# Only add header if the file is being created
			self.data_df.to_csv(f, sep='\t', header=f.tell()==0)

	def procrustes_transform(self, X):
		'''Transform embedding to match the reference frame of the first embedding matrix'''
		X_inv = np.linalg.pinv(X)
		T = np.dot(X_inv, self.X_0)
		u, s, v = np.linalg.svd(T, full_matrices=False)
		si = np.eye(u.shape[0], v.shape[0])
		T_rot = np.dot(u, np.dot(si, v))
		X_rot = np.dot(X, T_rot)
		return(X_rot)

