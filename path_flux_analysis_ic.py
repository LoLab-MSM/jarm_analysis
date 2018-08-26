from tropical.dominant_path_analysis import run_dompath_multi
from tropical.clustering import ClusterSequences
from tropical.plot_signatures import PlotSequences
from jnk3_no_ask1 import model
import numpy as np
import pickle

d = run_dompath_multi(model, 'simulations_ic_jnk3.h5', target='s27', depth=4, cpu_cores=20)
path_signatures = d['signatures']
np.save('dom_path_signatures.npy', path_signatures)

with open('dom_path_labels.pkl', 'wb') as f:
    pickle.dump(d['labels'], f, pickle.HIGHEST_PROTOCOL)

cs = ClusterSequences(path_signatures)
cs.diss_matrix(n_jobs=20)
sil_df = cs.silhouette_score_agglomerative_range(cluster_range=20, n_jobs=20)
nclusters = sil_df.loc[sil_df['cluster_silhouette'].idxmax()].num_clusters
nclusters = int(nclusters)
cs.spectral_clustering(nclusters=nclusters, n_jobs=20)

ps = PlotSequences(cs)
ps.plot_sequences(type_fig='modal', title='modal_path')
ps.plot_sequences(type_fig='trajectories', title='trajectories_path')
ps.plot_sequences(type_fig='entropy', title='modal_path')

