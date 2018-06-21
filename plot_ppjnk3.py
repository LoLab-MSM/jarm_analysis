from tropical.cluster_analysis import AnalysisCluster
from jnk3_no_ask1 import model

simulations = 'all_simulations_jnk3.h5'

a = AnalysisCluster(model, clusters=None, sim_results=simulations)

a.plot_dynamics_cluster_types([27], norm=False, fig_label='ppJNK3')