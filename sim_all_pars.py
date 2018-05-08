from jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from equilibration_function import pre_equilibration
from pathos.multiprocessing import ProcessingPool as Pool

tspan = np.linspace(0, 60, 60)
tspan_eq = np.linspace(0, 30, 30)

chain0 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_0_50000.npy')
chain1 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_1_50000.npy')
chain2 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_2_50000.npy')
chain3 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_3_50000.npy')
chain4 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_4_50000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations / 2)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :],
                          chain3[burnin:, :], chain4[burnin:, :]))

idx_pars_calibrate = [1, 15, 17, 19, 24, 25, 26, 27]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]
param_values = np.array([p.value for p in model.parameters])


def run_sim(parameters):
    param_values[rates_of_interest_mask] = 10 ** parameters
    pars_eq = np.copy(param_values)
    pars_eq[[24, 25]] = 0
    eq_conc = pre_equilibration(model, tspan_eq, parameters=pars_eq)[1]
    sim = ScipyOdeSimulator(model, tspan=tspan).run(param_values=param_values, initials=eq_conc).species
    return sim

p = Pool(25)
res = p.amap(run_sim, samples[4])
sims = res.get()
np.save('sim_results.npy', np.array(sims))