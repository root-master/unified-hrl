import pickle
# clustering results
results_file_path = './results/clustering_results_t_200000.pkl'
with open(results_file_path, 'rb') as f: 
	results = pickle.load(f)

import pickle
results_file_path ='./results/pickle_results_t_150000.pkl'
with open(results_file_path, 'rb') as f: 
	results = pickle.load(f)

max(results[2])

