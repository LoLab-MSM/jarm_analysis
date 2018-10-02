# coding=utf-8
from jnk3_no_ask1 import model
import numpy as np
import matplotlib.pyplot as plt
from tropical.util import get_simulations
import matplotlib.patches as mpatches
from tropical.visualize_discretization import visualization_path
import pickle
from tropical.util import path_differences
from tropical.cluster_analysis import AnalysisCluster as AC


plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels


def _get_observable(obs, y):
    from pysb.bng import generate_equations
    generate_equations(model)
    obs_names = [ob.name for ob in model.observables]
    try:
        obs_idx = obs_names.index(obs)
    except ValueError:
        raise ValueError(obs + "doesn't exist in the model")
    sps = model.observables[obs_idx].species
    obs_values = np.sum(y[:, :, sps], axis=2)
    return obs_values.T


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def plot_ic_dependence():
    # Initial condition used in simulations
    n_conditions = 5000
    max_arrestin = 100
    arrestin_initials = np.linspace(0, max_arrestin, n_conditions)
    arrestin_initials = arrestin_initials

    # obtain simulations
    trajectories, parameters, nsims, time = get_simulations('simulations/simulations_ic_scipy_box.h5')

    # Obtain species or observable to plot
    ppjnk3 = _get_observable('all_jnk3', trajectories)[-1]
    # ppjnk3 = np.array([s[-1][27] for s in trajectories])
    ppjnk3_max_idx = np.argmax(ppjnk3)
    print('max ppJNK3', max(ppjnk3))
    print('max arrestin', arrestin_initials[ppjnk3_max_idx])

    # Plot the values of ppJNK3 at the end of the simulations
    plt.semilogx(arrestin_initials, ppjnk3)

    # Fill areas under the curve given by the clusters
    clus_labels = np.load('ic_paths_analysis/depth7_scipy_05/path_labels_depth7_ic_box.npy')
    clus0_idxs = np.where(clus_labels == 0)[0]
    clus1_idxs = np.where(clus_labels == 1)[0]
    clus2_idxs = np.where(clus_labels == 2)[0]
    clus3_idxs = np.where(clus_labels == 3)[0]
    clus4_idxs = np.where(clus_labels == 4)[0]
    clus5_idxs = np.where(clus_labels == 5)[0]
    clus6_idxs = np.where(clus_labels == 6)[0]
    clus7_idxs = np.where(clus_labels == 7)[0]
    clus8_idxs = np.where(clus_labels == 8)[0]
    clus9_idxs = np.where(clus_labels == 9)[0]
    # clus10_idxs = np.where(clus_labels == 10)[0]
    # clus11_idxs = np.where(clus_labels == 11)[0]
    # clus12_idxs = np.where(clus_labels == 12)[0]

    colors = ['#FFFF00', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6',
              '#A30059', '#FFDBE5', "#7A4900"]  # , "#0000A6", "#63FFAC", "#B79762", "#004D43"]
    clus_idxs = [clus0_idxs, clus1_idxs, clus2_idxs, clus3_idxs, clus4_idxs, clus5_idxs,
                 clus6_idxs, clus7_idxs, clus8_idxs]  # , clus10_idxs, clus11_idxs, clus12_idxs]
    for clus in range(9):
        consecutive_idxs = consecutive(clus_idxs[clus])
        for consec in consecutive_idxs:
            plt.fill_between(arrestin_initials[consec], ppjnk3[consec], color=colors[clus])

    patches = [mpatches.Patch(color=colors[i], label='cluster{0}'.format(i)) for i in range(9)]
    plt.legend(handles=patches, loc=1)

    plt.axvline(arrestin_initials[ppjnk3_max_idx], color='r', linestyle='dashed', ymax=0.95)
    locs, labels = plt.xticks()
    locs = np.append(locs, arrestin_initials[ppjnk3_max_idx])
    plt.xticks(locs.astype(int))
    plt.xlim(0, 100)
    plt.xlabel(r'Arrestin [$\mu$M]')
    plt.ylabel(r'Activated JNK3 [$\mu$M]')
    plt.savefig('varying_arrestin_27_depth7_box.pdf', format='pdf', bbox_inches='tight')


def visualize_dom_graphs():
    with open("ic_paths_analysis/depth7_scipy_05/dom_path_labels_ic_box.pkl", "rb") as input_file:
        paths = pickle.load(input_file)

    visualization_path(model, paths[11], type_analysis='production', filename='ic_paths_analysis/depth7_scipy_05/path_11.pdf')
    # for keys, values in paths.items():
    #     visualization_path(model, values, 'path_{}.pdf'.format(keys))
    # print('1')
    # a=path_differences(model, paths)
    return #a


def plot_species_proportions():
    # ic_paths_analysis/depth7_scipy_05/path_labels_depth7_ic.npy
    # simulations/simulations_ic_scipy_box.h5

    # kpars_paths_analysis/depth_scipy_05/path_labels_depth7.npy
    # simulations/simulations_arrestin_jnk3_scipy.h5
    labels = np.load('ic_paths_analysis/depth7_scipy_05/path_labels_depth7_ic_box.npy')
    a = AC(model, 'simulations/simulations_ic_scipy_box.h5', clusters=labels)

    jnk3 = model.monomers['JNK3']
    mkk4 = model.monomers['MKK4']
    mkk7 = model.monomers['MKK7']
    arrestin = model.monomers['Arrestin']

    a.plot_pattern_sps_distribution(jnk3,  fig_name='jnk3_proportion',
                                    save_path='',
                                    type_fig='bar')

    # a.plot_pattern_rxns_distribution(jnk3, fig_name='jnk3_rxns_bar',
    #                                  type_fig='bar')


def check_kpars_simulations():
    # Plot the species dynamics of each cluster

    a = AC(model, 'simulations/simulations_arrestin_jnk3_scipy.h5', clusters=None)
    a.plot_cluster_dynamics(species=['all_jnk3'], fig_name='all_sims', norm=True, norm_value=0.6)