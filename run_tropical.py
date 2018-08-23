from tropical.pydrone import Pydrone
from jnk3_no_ask1 import model

pyd = Pydrone(model, 'simulations_ic_jnk3.h5', 1)
pyd.discretize(cpu_cores=20)
pyd.cluster_signatures_spectral(species='__s2_p', nclusters=10, cluster_range=True, cpu_cores=20)
ac = pyd.analysis_cluster
ps = pyd.plot_signatures
ac.plot_cluster_dynamics(27, fig_name='27')

# jnk3 monomer
jnk3 = model.monomers['JNK3']
ac.plot_pattern_rxns_distribution(jnk3, type_fig='bar', fig_name='jnk3')
ac.plot_pattern_rxns_distribution(jnk3, type_fig='entropy', fig_name='jnk3')
ac.plot_pattern_sps_distribution(jnk3, type_fig='bar', fig_name='jnk3')
ac.plot_pattern_sps_distribution(jnk3, type_fig='entropy', fig_name='jnk3')

ps.plot_sequences(type_fig='modal', title='jnk3')
ps.plot_sequences(type_fig='trajectories', title=jnk3)
