from tropical.pydrone import Pydrone
from jnk3_no_ask1 import model

pyd = Pydrone(model, 'simulations_ic_jnk3.h5', 1)
print('loaded simulations')
pyd.discretize(cpu_cores=20)
print('Discretized trajectories')
pyd.cluster_signatures_spectral(species='__s1_c', nclusters=10, cluster_range=True, cpu_cores=20)
print('clustered trajectories')
ac = pyd.analysis_cluster
ps = pyd.plot_signatures

print('Plotting')
ac.plot_cluster_dynamics([1], fig_name='1', norm=True, norm_value=0.05)

# jnk3 monomer
jnk3 = model.monomers['JNK3']
mkk4 = model.monomers['MKK4']
pattern = mkk4

ac.plot_pattern_rxns_distribution(pattern, type_fig='bar', fig_name='bar_{0}'.format(pattern.name))
ac.plot_pattern_rxns_distribution(pattern, type_fig='entropy', fig_name='ent_{0}'.format(pattern.name))
ac.plot_pattern_sps_distribution(pattern, type_fig='bar', fig_name='bar_{0}'.format(pattern.name))
ac.plot_pattern_sps_distribution(pattern, type_fig='entropy', fig_name='ent_{0}'.format(pattern.name))

ps.plot_sequences(type_fig='modal', title='modal')
ps.plot_sequences(type_fig='trajectories', title='trajectories')
