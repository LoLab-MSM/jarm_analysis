from jnk3_no_ask1 import model
import numpy as np
import scipy.stats
from tropical.util import get_simulations
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels

chain0 = np.load('pydream_results_test/jnk3_dreamzs_5chain_sampled_params_chain_0_500000.npy')
chain1 = np.load('pydream_results_test/jnk3_dreamzs_5chain_sampled_params_chain_1_500000.npy')
chain2 = np.load('pydream_results_test/jnk3_dreamzs_5chain_sampled_params_chain_2_500000.npy')
chain3 = np.load('pydream_results_test/jnk3_dreamzs_5chain_sampled_params_chain_3_500000.npy')
chain4 = np.load('pydream_results_test/jnk3_dreamzs_5chain_sampled_params_chain_4_500000.npy')

chain0_logp = np.load('pydream_results_test/jnk3_dreamzs_5chain_logps_chain_0_500000.npy')
chain1_logp = np.load('pydream_results_test/jnk3_dreamzs_5chain_logps_chain_1_500000.npy')
chain2_logp = np.load('pydream_results_test/jnk3_dreamzs_5chain_logps_chain_2_500000.npy')
chain3_logp = np.load('pydream_results_test/jnk3_dreamzs_5chain_logps_chain_3_500000.npy')
chain4_logp = np.load('pydream_results_test/jnk3_dreamzs_5chain_logps_chain_4_500000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations / 2)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :],
                          chain3[burnin:, :], chain4[burnin:, :]))

samples_logp = np.concatenate((chain0_logp[burnin:, :], chain1_logp[burnin:, :], chain2_logp[burnin:, :],
                          chain3_logp[burnin:, :], chain4_logp[burnin:, :]))
# Save most likely parameter
unique_pars, indices, counts = np.unique(samples, return_index=True, return_counts=True, axis=0)
counts_sorted = np.argsort(counts)[::-1]

# plt.hist(np.exp(-samples_logp), bins=50)
# plt.xlabel('-ln(posterior)')
# plt.ylabel('No. Parameter vectors')
# plt.show()

sims_arrestin_5000, _, _, _ = get_simulations('simulations/simulations_arrestin_scipy_181.h5')
sims_noarrestin_5000, _, _, _ = get_simulations('simulations/simulations_arrestin_scipy_181.h5')

exp_data = pd.read_csv('data/exp_data_3min.csv')
tspan = np.linspace(0, exp_data['Time (secs)'].values[-1], 181)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:] for idx in tspan]

def get_observable(obs, y):
    from pysb.bng import generate_equations
    generate_equations(model)
    obs_names = [ob.name for ob in model.observables]
    try:
        obs_idx = obs_names.index(obs)
    except ValueError:
        raise ValueError(obs + "doesn't exist in the model")
    sps = model.observables[obs_idx].species
    obs_values = np.sum(y[:, :, sps], axis=2)
    return obs_values

ptyr_sim_5000 = get_observable('pTyr_jnk3', sims_arrestin_5000)
pthr_sim_5000 = get_observable('pThr_jnk3', sims_arrestin_5000)
ptyr_noarr_sim_5000 = get_observable('pTyr_jnk3', sims_noarrestin_5000)
pthr_noarr_sim_5000 = get_observable('pThr_jnk3', sims_noarrestin_5000)


def cost(ptyr_sim, pthr_sim, ptyr_noarr_sim, pthr_noarr_sim):

    e1 = np.sum((exp_data['pTyr_arrestin_avg'].values - ptyr_sim[t_exp_mask])**2 )

    e2 = np.sum((exp_data['pThr_arrestin_avg'].values - pthr_sim[t_exp_mask]) ** 2)

    e3 = np.sum((exp_data['pTyr_noarrestin_avg'].values - ptyr_noarr_sim[t_exp_mask]) ** 2)

    e4 = np.sum((exp_data['pThr_noarrestin_avg'].values - pthr_noarr_sim[t_exp_mask]) ** 2)

    total_error = e1 + e2 + e3 + e4
    return total_error


all_errors = [cost(ptyr_sim_5000[i], pthr_sim_5000[i], ptyr_noarr_sim_5000[i],
                   pthr_noarr_sim_5000[i]) for i in range(5000)]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


print(mean_confidence_interval(all_errors))

plt.hist(all_errors, bins=15)
plt.xlim((0, 5))
plt.xlabel('Sum of Squared Errors')
plt.ylabel('Counts')
plt.savefig('goodness_of_fit.pdf', format='pdf', bbox_inches='tight')