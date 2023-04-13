"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
from plot_policy import convert_res_dict
import pickle
import os

def plot_multi_beta(res_files):
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(1,2,figsize=(10,5))

    params = [0.05, 0.1, 0.5, 1.0]
    for param in params:
        xcoods = np.linspace(beta.ppf(0.01, param, param), beta.ppf(0.99, param, param), 100)
        ycoods = beta.pdf(xcoods, param, param)

        label = f"Beta ({param}, {param})" + (" [=U(0, 1)]" if param == 1. else "")
        axes[0].plot(xcoods, ycoods, label=label)

        # find correct res file
        rfiles = [e for e in res_files if str(param) in e]
        if len(rfiles) != 1:
            continue
        rfile = rfiles[0]
        res = pickle.load(open(rfile, "rb"))
        res = convert_res_dict(res)

        # collect data
        keys, returns, stds, dists = [], [], [], []
        for key in res.keys():
            keys.append(key)
            returns.append(res[key]["return"][0])
            stds.append(res[key]["return"][1])
            dists.append(res[key]["dist. to org."][0])

        # plot   
        means = np.array(returns) / 100
        stds = np.array(stds) / 100
        axes[1].plot(keys, means)
        axes[1].fill_between(keys, means - stds, means + stds, alpha=0.5)

        print(param, dists)

    axes[0].set_ylim((0., 3.))
    axes[0].set_title("Beta PDFs")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("p(x)")
    plt.setp(axes[0].get_yticklabels(), visible=False)
    axes[0].tick_params(axis='y', colors='w')

    axes[1].set_title("LION performances")
    axes[1].set_xlabel("$\lambda$")
    axes[1].set_ylabel("Evaluation Return")
    line = axes[1].axhline(-286, linestyle="dashed", color="black", label="behavior performance")
    axes[1].legend()
    line.set_label(None)

    axes[0].grid(linestyle="dotted")
    axes[0].set_facecolor((0.95, 0.95, 0.95))

    axes[1].grid(linestyle="dotted")
    axes[1].set_facecolor((0.95, 0.95, 0.95))

    fig.legend(loc=(0.025, 0.025), ncol=len(params))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig("results/betas.png")


if __name__ == "__main__":
    plot_multi_beta([
        "results/repeat_bad02_bathtub_0.05.p",
        "results/repeat_bad02_bathtub_0.1.p",
        "results/repeat_bad02_bathtub_0.5.p",
        "results/repeat_bad02_bathtub_1.0.p",
    ])
