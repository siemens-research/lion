"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import matplotlib.pyplot as plt
from plot_policy import convert_res_dict
import pickle

def plot_multi_eta(res_files):
    plt.rcParams.update({'font.size': 13})
    fig, axis = plt.subplots(1,1,figsize=(6,5))

    etas = [float(r.split("_")[-1].split(".p")[0]) for r in res_files]
    data_dists = []
    returns = []

    for rfile in res_files:
        res = pickle.load(open(rfile, "rb"))
        res = convert_res_dict(res)
        res = res[0.0]
        #data_dists.append(res["dist_data_euclid"])
        data_dists.append(res["dist. to org."][0])
        returns.append(res["return"][0] / 100)

    tuples = [(etas[i], data_dists[i], returns[i]) for i in range(len(etas))]
    tuples = sorted(tuples, key=lambda x: x[0])
    xs = [t[0] for t in tuples]
    ys = [t[1] for t in tuples]
    ys2 = [t[2] for t in tuples]
    axis.plot(xs, ys)

    #ax2 = axis.twinx()
    #ax2.plot(xs, ys2, color="green")
    #ax2.set_ylabel("Performance")

    #axes[0].set_ylim((0., 3.))
    axis.set_title("Beta Distribution Impact @ $\lambda=0$")
    #axis.set_xlabel("$\eta$")
    axis.set_xlabel("x in Beta(x,x)")
    #axis.set_ylabel("Data Distance")
    axis.set_ylabel("Distance to Original Policy Model")
    #plt.setp(axes[0].get_yticklabels(), visible=False)
    #axes[0].tick_params(axis='y', colors='w')

    plt.tight_layout()
    #fig.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig("results/beta_data.png")
    plt.clf()


if __name__ == "__main__":
    # plot_multi_eta([
    #     "single_ress_IB/bad02_eta_0.0.p",
    #     "single_ress_IB/bad02_eta_0.1.p",
    #     "single_ress_IB/bad02_eta_0.2.p",
    #     "single_ress_IB/bad02_eta_0.3.p",
    #     "single_ress_IB/bad02_eta_0.4.p",
    #     "single_ress_IB/bad02_eta_0.5.p",
    #     "single_ress_IB/bad02_eta_0.6.p",
    #     "single_ress_IB/bad02_eta_0.7.p",
    #     "single_ress_IB/bad02_eta_0.8.p",
    #     "single_ress_IB/bad02_eta_0.9.p",
    #     "single_ress_IB/bad02_eta_1.0.p",
    # ])
    plot_multi_eta([
        "results/repeat_bad02_bathtub_0.05.p",
        "results/repeat_bad02_bathtub_0.1.p",
        "results/repeat_bad02_bathtub_0.5.p",
        "results/repeat_bad02_bathtub_1.0.p",
    ])
