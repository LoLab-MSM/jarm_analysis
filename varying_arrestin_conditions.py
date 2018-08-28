# coding=utf-8
from model_analysis.jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt
from model_analysis.equilibration_function import pre_equilibration
from tropical.util import get_simulations

plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
#New kds in jnk3 mkk4/7
# idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream
# idx_pars_calibrate = [5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream2
idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

calibrated_pars = np.load('most_likely_par_100000.npy')
param_values = np.array([p.value for p in model.parameters])

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars

tspan = np.linspace(0, 60, 100)

n_conditions = 5000
max_arrestin = 100
arrestin_initials = np.linspace(0, max_arrestin, n_conditions)
arrestin_initials = arrestin_initials
par_clus1 = par_set_calibrated

arrestin_idx = 44
kcat_idx = [36, 37]

repeated_parameter_values = np.tile(par_clus1, (n_conditions, 1))
repeated_parameter_values[:, arrestin_idx] = arrestin_initials
np.save('arrestin_diff_IC_par0.npy', repeated_parameter_values)

time_eq = np.linspace(0, 1000, 100)
pars_ic_eq = np.copy(repeated_parameter_values)
pars_ic_eq[:, kcat_idx] = 0  # Setting catalytic reactions to zero for pre-equilibration
# eq_conc = pre_equilibration(model, time_eq, pars_ic_eq)[1]

# sim1 = ScipyOdeSimulator(model=model, tspan=tspan).run(param_values=repeated_parameter_values, initials=eq_conc).all
trajectories, parameters, nsims, time = get_simulations('simulations_ic_jnk3.h5')
print(trajectories.shape)
ppjnk3 = np.array([s[27][-1] for s in trajectories])
ppjnk3_max_idx = np.argmax(ppjnk3)

# Plot the values of ppJNK3 at the end of the simulations
plt.plot(arrestin_initials, ppjnk3)

# Fill areas under the curve given by the clusters
labels = np.load('path_labels.npy')
clus0_idxs = np.where(labels == 0)
clus1_idxs = np.where(labels == 1)
clus2_idxs = np.where(labels == 2)
clus3_idxs = np.where(labels == 3)

plt.fill_between(arrestin_initials[clus0_idxs], ppjnk3[clus0_idxs], color='#E69F00')
plt.fill_between(arrestin_initials[clus1_idxs], ppjnk3[clus1_idxs], color='#0072B2')
plt.fill_between(arrestin_initials[clus2_idxs], ppjnk3[clus2_idxs], color='#CC79A7')
plt.fill_between(arrestin_initials[clus3_idxs], ppjnk3[clus3_idxs], color='#009E73')

plt.annotate('Cluster 2',
            xy=(7, 0.03), xycoords='data',
            xytext=(0.6, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

plt.annotate('Cluster 0',
            xy=(40, 0.03), xycoords='data',
            xytext=(0.8, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

plt.annotate('Cluster 1',
            xy=(80, 0.02), xycoords='data',
            xytext=(1, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

plt.annotate('Cluster 3',
            xy=(2, 0.04), xycoords='data',
            xytext=(0.4, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')


plt.axvline(arrestin_initials[ppjnk3_max_idx], color='r', linestyle='dashed', ymax=0.95)
locs, labels = plt.xticks()
locs = np.append(locs, arrestin_initials[ppjnk3_max_idx])
plt.xticks(locs.astype(int))
plt.xlim(0, max_arrestin)
plt.xlabel(r'Arrestin [$\mu$M]')
plt.ylabel(r'Activated JNK3 [$\mu$M]')

plt.savefig('varying_arrestin_27.pdf', format='pdf', bbox_inches='tight')
# plt.show()