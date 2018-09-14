from jnk3_no_ask1 import model
import numpy as np
from equilibration_function import pre_equilibration
from pysb.simulator import CupSodaSimulator

chain0 = np.load('pydream_results1/jnk3_dreamzs_5chain_sampled_params_chain_0_500000.npy')
chain1 = np.load('pydream_results1/jnk3_dreamzs_5chain_sampled_params_chain_1_500000.npy')
chain2 = np.load('pydream_results1/jnk3_dreamzs_5chain_sampled_params_chain_2_500000.npy')
chain3 = np.load('pydream_results1/jnk3_dreamzs_5chain_sampled_params_chain_3_500000.npy')
chain4 = np.load('pydream_results1/jnk3_dreamzs_5chain_sampled_params_chain_4_500000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations / 2)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :],
                          chain3[burnin:, :], chain4[burnin:, :]))

idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43]  # pydream3

# Save most likely parameter
unique_pars, indices, counts = np.unique(samples, return_index=True, return_counts=True, axis=0)
max_idx = np.argsort(counts)[::-1]

rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

param_values = np.array([p.value for p in model.parameters])

par_set_calibrated = np.copy(param_values)
all_par_set_calibrated = np.tile(par_set_calibrated, (len(unique_pars), 1))
all_par_set_calibrated[:, rates_of_interest_mask] = 10 ** unique_pars

tspan = np.linspace(0, 60, 100)

arrestin_idx = 44
kcat_idx = [36, 37]

tspan_eq = np.linspace(0, 100, 100)
integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}

cupsoda_solver = CupSodaSimulator(model, tspan=tspan_eq, gpu=0, obs_species_only=False,
                                  memory_usage='shared_constant', integrator_options=integrator_opt)

pars_eq = np.copy(all_par_set_calibrated)
pars_eq[:, kcat_idx] = 0
conc_eq = pre_equilibration(cupsoda_solver, param_values=pars_eq)[1]

sims_final = CupSodaSimulator(model, tspan=tspan, gpu=0, obs_species_only=False,
                              memory_usage='shared_constant',
                              integrator_options=integrator_opt).run(param_values=all_par_set_calibrated,
                                                                     initials=conc_eq)

sims_final.save('simulations_arrestin_jnk3.h5')

# Simulations without arrestin
jnk3_initial_idxs = [47, 48, 49]

pars_eq_noarrestin = np.copy(all_par_set_calibrated)
all_pars_noarrestin = np.copy(all_par_set_calibrated)
pars_eq_noarrestin[:, arrestin_idx] = 0
all_pars_noarrestin[:, arrestin_idx] = 0
pars_eq_noarrestin[:, jnk3_initial_idxs] = [0.592841488, 0, 0.007158512]

pars_eq_noarrestin[:, kcat_idx] = 0
conc_eq_noarrestin = pre_equilibration(cupsoda_solver, param_values=pars_eq_noarrestin)[1]

sims_final_noarrestin = CupSodaSimulator(model, tspan=tspan, gpu=0, obs_species_only=False,
                                         memory_usage='shared_constant',
                                         integrator_options=integrator_opt).run(param_values=all_pars_noarrestin,
                                                                                initials=conc_eq_noarrestin)
sims_final_noarrestin.save('simulations_noarrestin_jnk3.h5')
