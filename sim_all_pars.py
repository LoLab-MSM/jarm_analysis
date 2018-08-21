from jnk3_no_ask1 import model
import numpy as np
from equilibration_function import pre_equilibration
from pysb.simulator import CupSodaSimulator

tspan = np.linspace(0, 60, 60)
tspan_eq = np.linspace(0, 30, 30)

chain0 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_0_200000.npy')
chain1 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_1_200000.npy')
chain2 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_2_200000.npy')
chain3 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_3_200000.npy')
chain4 = np.load('calibration_normalization_preequilibration/jnk3_dreamzs_5chain_sampled_params_chain_4_200000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations / 2)
print(burnin)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :],
                          chain2[burnin:, :],
                          chain3[burnin:, :],
                          chain4[burnin:, :]))

idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27,
                      29, 33, 37, 38, 39, 40, 41, 43, 45]
rates_of_interest_mask = [i in idx_pars_calibrate
                          for i, par in enumerate(model.parameters)]
param_values = np.array([p.value for p in model.parameters])

all_parameters = np.tile(param_values, (burnin * 5, 1))
print(all_parameters[:, idx_pars_calibrate].shape)
print(samples.shape)
all_parameters[:, idx_pars_calibrate] = 10 ** samples

integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}

cupsoda_solver = CupSodaSimulator(model, tspan=tspan_eq, gpu=0,
                                  obs_species_only=False,
                                  memory_usage='shared_constant',
                                  integrator_options=integrator_opt)

kcat_idx = [38, 39]

pars_eq = np.copy(all_parameters)
pars_eq[:, kcat_idx] = 0
conc_eq = pre_equilibration(cupsoda_solver, param_values=pars_eq)[1]

sims_final = CupSodaSimulator(model, tspan=tspan, gpu=0, obs_species_only=False,
                              memory_usage='shared_constant',
                              integrator_options=integrator_opt).run(param_values=all_parameters, initials=conc_eq)

sims_final.save('all_simulations_jnk3.h5')
