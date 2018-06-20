from tropical import clustering
import pickle
import numpy as np

with open('pydream_kpars_signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

cpus = 30
rxn_type = 'consumption'

uujnk3_signatures = all_signatures[3][rxn_type]
clus = clustering.ClusterSequences(seqdata=uujnk3_signatures, unique_sequences=False)
# diss = np.load('pydream_uujnk3_diss.npy')
# clus.diss = diss
clus.diss_matrix(n_jobs=cpus)
np.save('pydream_uujnk3_diss.npy', clus.diss)

sil_df = clus.silhouette_score_spectral_range(cluster_range=range(2, 20), n_jobs=cpus, random_state=1234)
print (sil_df)
best_silh, n_clus = sil_df.loc[sil_df['cluster_silhouette'].idxmax()]
n_clus = int(n_clus)
# n_clus = 7

clus.spectral_clustering(n_clusters=n_clus, n_jobs=cpus, random_state=1234)
np.save('pydream_labels_uujnk3.npy', clus.labels)
b = clustering.PlotSequences(clus)
b.modal_plot(title='uuJNK3')
b.all_trajectories_plot(title='uuJNK3')

