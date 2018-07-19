from jnk3_no_ask1 import model
import numpy as np
from equilibration_function import pre_equilibration
from pysb.simulator import CupSodaSimulator

idx_pars_calibrate = [1, 15, 17, 19, 24, 25, 26, 27]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

calibrated_pars = np.load('pydream_most_likely_par.npy')
param_values = np.array([p.value for p in model.parameters])

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars

tspan = np.linspace(0, 60, 100)

n_conditions = 10000
max_arrestin = 100
arrestin_initials = np.linspace(0, max_arrestin, n_conditions)
par_clus1 = par_set_calibrated

repeated_parameter_values = np.tile(par_clus1, (n_conditions, 1))
repeated_parameter_values[:, 32] = arrestin_initials
np.save('arrestin_diff_IC_par0.npy', repeated_parameter_values)

tspan_eq = np.linspace(0, 30, 30)
integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}

cupsoda_solver = CupSodaSimulator(model, tspan=tspan_eq, gpu=0, obs_species_only=False,
                                  memory_usage='shared_constant', integrator_options=integrator_opt)

pars_eq = np.copy(repeated_parameter_values)
pars_eq[:, [24, 25]] = 0
conc_eq = pre_equilibration(cupsoda_solver, param_values=pars_eq)[1]

sims_final = CupSodaSimulator(model, tspan=tspan, gpu=0, obs_species_only=False,
                              memory_usage='shared_constant',
                              integrator_options=integrator_opt).run(param_values=repeated_parameter_values,
                                                                     initials=conc_eq)

sims_final.save('simulations_ic_jnk3.h5')
