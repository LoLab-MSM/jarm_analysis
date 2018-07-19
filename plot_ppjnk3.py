from tropical.cluster_analysis import AnalysisCluster
from jnk3_no_ask1 import model
import numpy as np

simulations = 'simulations_ic_jnk3_oneP.h5'
labels = np.load('ic_labels_pro1p_alljnk3.npy')

a = AnalysisCluster(model, sim_results=simulations, clusters=labels)

a.hist_avg_sps(model.monomers['JNK3'], fig_name='JNK3')
#a.plot_dynamics_cluster_types([model.reactions_bidirectional[26]['rate']], norm=False)
#a.plot_sp_ic_overlap([32])


