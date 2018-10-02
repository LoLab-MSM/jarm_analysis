import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from jnk3_no_ask1 import model

chain0 = np.load('pydream_results_test1/jnk3_dreamzs_5chain_sampled_params_chain_0_500000.npy')
chain1 = np.load('pydream_results_test1/jnk3_dreamzs_5chain_sampled_params_chain_1_500000.npy')
chain2 = np.load('pydream_results_test1/jnk3_dreamzs_5chain_sampled_params_chain_2_500000.npy')
chain3 = np.load('pydream_results_test1/jnk3_dreamzs_5chain_sampled_params_chain_3_500000.npy')
chain4 = np.load('pydream_results_test1/jnk3_dreamzs_5chain_sampled_params_chain_4_500000.npy')

total_iterations = chain0.shape[0]
burnin = int(total_iterations / 2)
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :],
                          chain3[burnin:, :], chain4[burnin:, :]))


def plot_pars_dist():
    # Plots and saves the marginal distributions of the fitted model parameters

    # idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream
    # idx_pars_calibrate = [5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream2
    idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43]  # pydream3

    ndims = len(idx_pars_calibrate)
    colors = sns.color_palette(n_colors=ndims)
    rows = 6
    columns = 3
    counter = 0

    # Save most likely parameter
    u, indices, counts = np.unique(samples, return_index=True, return_counts=True, axis=0)
    max_idx = np.argsort(counts)[::-1]
    # np.save('most_likely_par_100000.npy', u[max_idx])

    f, axes = plt.subplots(rows, columns, figsize=(7, 7), sharex=True)
    for r in range(rows):
        for c in range(columns):
            weights = np.ones_like(samples[:, counter]) / float(len(samples[:, counter]))
            # axes[r, c].vlines(u[max_idx[:50]][:, counter], ymin=0, ymax=0.01, color='k', linestyle='solid', linewidth=0.0001)
            axes[r, c].hist(samples[:, counter], bins=25, color=colors[counter], weights=weights)
            axes[r, c].set_title(model.parameters[idx_pars_calibrate[counter]].name, fontdict={'fontsize': 8})
            # axes[r, c].set_xlim(-6, 6)
            counter += 1

            if counter > len(idx_pars_calibrate):
                break
    f.add_subplot(111, frameon=False)
    f.subplots_adjust(hspace=0.5)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Log(Parameter value)", fontsize=14)
    plt.ylabel("Probability", fontsize=14, labelpad=15)
    # plt.show()
    plt.savefig('pars_dist_plot_box.pdf', format='pdf', bbox_inches="tight")


def most_likely_par():
    # saves the most likely parameter from the posterior distribution
    u, indices, counts = np.unique(samples, return_index=True, return_counts=True, axis=0)
    max_idx = np.argmax(counts)
    np.save('most_likely_par_500000_box.npy', u[max_idx])


