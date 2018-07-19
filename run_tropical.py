import pickle
from tropical.dynamic_signatures_range import run_tropical_multi
from jnk3_no_ask1 import model

a = run_tropical_multi(model, simulations='simulations_ic_jnk3.h5', cpu_cores=30, add_observables=True, verbose=True)

with open('ic_obs_signatures.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)