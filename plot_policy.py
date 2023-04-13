"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import os
import math
import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from load_data import get_lsy_data

BASEPATH = None

def plot_policy(actor, savepath, rows=7, cols=7, lamdas=np.arange(49), envname="simpleenv"):
    fig, axes = plt.subplots(rows, cols, figsize=(40, 20))
    if rows==1: 
        try:
            axes = axes.reshape((1, -1))
        except:
            axes = np.array(axes).reshape((1, 1))
    lamdas = iter(lamdas)

    dim_lim = 10

    for i in range(rows):
        for j in range(cols):

            states_grid = np.vstack([[x,y] for x in np.arange(0, dim_lim + 0.5, 0.5) for y in np.arange(0, dim_lim + 0.5, 0.5)])
            states_grid = torch.tensor(states_grid.astype(np.float32))

            try:
                lamda_scalar = round(next(lamdas), 2)
            except:
                break
            lamda = np.zeros(states_grid.shape[0]).reshape((-1, 1)).astype(np.float32) + lamda_scalar
            lamda = torch.Tensor(lamda)
            
            actions_grid = actor.act(states_grid, lamda)
            actions_grid = actions_grid.detach().numpy()
            
            action_norms = np.linalg.norm(actions_grid, axis=1).reshape((-1, 1))
            actions_normed = actions_grid / action_norms

            angles = np.arccos(actions_normed[:,0])  # those are the angles to (1,0) - no matter which direction
            angles[actions_normed[:,1] < 0] = math.pi + (math.pi - angles[actions_normed[:,1] < 0])  # adjusted for direction
            dsize = int(np.sqrt(angles.shape[0]))
            angles = angles.reshape((dsize, dsize))
            angles = angles.T
            angles = np.flip(angles, 0)

            axes[i,j].set_title(f"Policymap - $\lambda = {lamda_scalar}$")
            im = axes[i,j].imshow(angles, cmap="jet", vmin=0, vmax=2*math.pi, extent=[0,dim_lim,0,dim_lim])
            axes[i,j].quiver(states_grid[:, 0], states_grid[:, 1], actions_grid[:, 0], actions_grid[:, 1])

            axes[i,j].set_xticks([0, 2, 4, 6, 8, 10])
            if j == 0:
                axes[i,j].set_yticks([0, 2, 4, 6, 8, 10])
            else:
                axes[i,j].set_yticks([])

            islast = (j == cols - 1)
            #islast = False
            if islast:
                #divider = make_axes_locatable(axes[i,j])
                #cax = divider.append_axes("right", size="5%", pad=0.1)
                cax = fig.add_axes([axes[i,j].get_position().x1+0.01,axes[i,j].get_position().y0,0.01,axes[i,j].get_position().y1-axes[i,j].get_position().y0])
                cbar = plt.colorbar(im, cax=cax, label='angle to (1,0)')
                cbar.ax.get_yaxis().set_ticks([0., 0.5*math.pi, 1.0*math.pi, 1.5*math.pi, 2.0*math.pi])
                cax.set_yticklabels(["$0$", "$0.5\pi$", "$1.0\pi$", "$1.5\pi$", "$2.0\pi$"])

    #plt.tight_layout()
    plt.savefig(savepath + ".pdf", bbox_inches='tight')
    fig.clear()
    plt.close(fig)


class MultiPolicy():
    def __init__(self, policy_dict):
        self.policies = policy_dict

    def act(self, state, lamda):
        single_lamda = lamda[0,0].item()
        agent = self.policies[single_lamda]
        print(type(agent), type(self.policies))
        return agent.act(state, lamda)


def convert_res_dict(result):
    if type(result[list(result.keys())[0]]) == dict:
        return result
    
    new_result = {}
    for key in result.keys():
        single_res = result[key]
        new_single = {
            "return": (single_res[0], single_res[1]),
            "dist. to org.": (single_res[6], single_res[7]),
        }
        new_result[key] = new_single
    
    return new_result


def get_real_distr(dsname):
    assert BASEPATH is not None, "You need to specify the data directory, i.e. /path/to/industrial_benchmark/datasets/"
    setting, value = dsname.split("lsy-")[-1].split("-")
    basepath = BASEPATH + setting + "/"
    fname = [x for x in os.listdir(basepath) if value in x][0]
    fpath = basepath + fname
    data, _ = get_lsy_data(fpath, 1.0)
    states = data["obs"]
    distr = states[:, 1:4].mean(axis=0), states[:, 1:4].std(axis=0), states[:, 1:4].min(axis=0), states[:, 1:4].max(axis=0)
    distdict = {
        "velocity_distr": tuple(distr[i][0] for i in range(len(distr))),
        "gain_distr": tuple(distr[i][1] for i in range(len(distr))),
        "shift_distr": tuple(distr[i][2] for i in range(len(distr))),
    }
    return distdict


def plot_performances(pfiles, savepath, extra=None):
    plt.rcParams.update({'font.size': 16})
    new_savepath = savepath + ".png"
    pfiles = [x for x in pfiles if x != "config.p"]    
    fig = plt.figure(figsize=(6, 5)) # 30

    ratios_single = [2, 1]

    if len(pfiles) > 6:
        fig_columns = 6
        fig_rows = 3
    else:
         fig_columns = len(pfiles)
         fig_rows = 1

    ratios = ratios_single * fig_rows
    fig_rows = len(ratios)

    labels2colors = {
                "return": "blue",
                "dist_real_euclid": "orange",
                "dist_real_maha": "red",
                "dist. to org.": "orange",
                "dist_learn_maha": "red",
                "dist_data_euclid": "orange",
                "dist_data_maha": "red",
                "velocity_distr": "green",
                "gain_distr": "red",
                "shift_distr": "blue",
                "unknown_visits": "purple",
            }

    gs = gridspec.GridSpec(fig_rows, fig_columns, height_ratios=ratios) # , hspace=0.05 # not compatible with tight_layout

    extra_ = False
    for i, filelist in enumerate([pfiles, extra]):
        if i == 1:
            extra_ = True
            if extra is None:
                continue

        for k, pfile in enumerate(filelist):
            if pfile == "": continue
            res = pickle.load(open(pfile, "rb"))
            res = convert_res_dict(res)
            keys = np.array([key for key in res.keys()])
            dsname = pfile.split(".p")[0].split("/")[-1]

            ylabels = list(res[keys[0]].keys())
            rows = len(ylabels)
            
            #fig = plt.figure(figsize=(10,25))
            starting_index = k % fig_columns + (k - (k%fig_columns)) * 2
            ax0 = plt.subplot(gs[starting_index])
            axes = [plt.subplot(gs[starting_index+(j*fig_columns)], sharex = ax0) for j in range(1, rows)]
            axes = [ax0] + axes
            axes[-1].set_xlabel("$\lambda$ - Optimization Mixture Parameter")

            real_dist = get_real_distr(dsname)
            
            for i, lab in enumerate(ylabels):
                if type(res[keys[0]][lab]) != tuple:
                    means = np.array([res[key][lab] for key in res.keys()])
                    axes[i].plot(keys, means, color=labels2colors[lab])

                else:
                    if len(res[keys[0]][lab]) == 2:
                        means = np.array([res[key][lab][0] for key in res.keys()])
                        stds = np.array([res[key][lab][1] for key in res.keys()])
                        desc = None
                        if lab == "return":
                            means /= 100
                            stds /= 100
                            desc = "LION (ours)" if k == len(filelist)-1 else None
                            if extra_:
                                desc = "$\lambda$-TD3+BC" if k == len(filelist)-1 else None
                        elif lab == "dist. to org.":
                            desc = r"LION (distance to original)" if k == 0 else None
                            desc = None

                        if extra_:
                            color = "cyan"
                        else:
                            if extra is not None:
                                color = "blue"
                            else:
                                color = labels2colors[lab]
                        axes[i].fill_between(keys, means - stds, means + stds, alpha=0.5, color=color)
                        axes[i].plot(keys, means, color=color, label=desc)

                    else:
                        mins = np.array([res[key][lab][2] for key in res.keys()])
                        maxs = np.array([res[key][lab][3] for key in res.keys()])

                        axes[i].plot(keys, mins, color=labels2colors[lab], linestyle="dashed")
                        axes[i].plot(keys, maxs, color=labels2colors[lab], linestyle="dashed")

                        #comparison
                        axes[i].plot(keys, [real_dist[lab][2]]*len(keys), color="yellow", linestyle="dotted")
                        axes[i].plot(keys, [real_dist[lab][3]]*len(keys), color="yellow", linestyle="dotted")

                if k == 0:
                    axes[i].set_ylabel(lab)

                if lab == "return":
                    titlestring = " ".join(dsname.split("_lsy")[1].split("-")) if "1.0" not in dsname else "global 1.0"
                    axes[i].set_title("IB " + titlestring)
                    acolors = ["red", "green", "yellow", "grey", "purple", "cyan", "brown", "black", "pink"]
                    algos = ["MOOSE", "WSBC", "BRAC-v", "BEAR", "BCQ", "TD3+BC", "CQL", "MOPO", "MOReL (below Y-range)"]
                    values = dsname2baseline[dsname]

                    if extra is None:
                        for value, algo, acolor in zip(values, algos, acolors):
                            thelabel = algo if k == 0 else None
                            axes[i].axhline(value, label=thelabel, color=acolor, linestyle="dashed")

                        lowerbound = min(means) - 0.05 * abs(min(means))
                        upperbound = max(means) + 0.05 * abs(min(means))
                        axes[i].set_ylim((lowerbound, upperbound))

                    if i != len(ylabels) - 1:
                        plt.setp(axes[i].get_xticklabels(), visible=False)
                
                    if k == len(filelist) - 1 and extra_:
                        axes[i].legend(loc="center right")

                axes[i].grid(linestyle="dotted")
                axes[i].set_facecolor((0.95, 0.95, 0.95))

    if extra_:
        fig.subplots_adjust(bottom=0.15, top=0.9, hspace=0.05)
        fig.tight_layout()
    else:
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25, top=0.9)
        fig.legend(loc=(0.01, 0.025), ncol=len(algos) + 2)
        plt.subplots_adjust(hspace=0.05)

    plt.savefig(new_savepath) # bbox_inches='tight'
    plt.clf()

# experiment results from other algos...
dsname2baseline = {
    "IB_lsy-bad-0.0": (-311., -134., -274., -322., -313., -325., -291., -123., -144.),
    "IB_lsy-bad-0.2": (-128., -118., -270., -168., -281., -289., -327., -110., -326.),
    "IB_lsy-bad-0.4": (-110., -103., -199., -129., -234., -230., -326., -139., -326.),
    "IB_lsy-bad-0.6": (-92.7, -84.9, -188., -90., -127., -172., -322., -130., -327.),
    "IB_lsy-bad-0.8": (-71.3, -70.0, -140., -90., -89., -112., -271., -119., -327.),
    "IB_lsy-bad-1.0": (-64.1, -63.8, -113., -65.1, -68.6, -65.4, -66.4, -105.1, -326.8),

    "IB_lsy-mediocre-0.0": (-83.3, -71.1, -117., -111., -105., -79.7, -88.9, -102.4, -121.8),
    "IB_lsy-mediocre-0.2": (-76.6, -68.5, -98.3, -115., -77.1, -77.7, -80.5, -119.1, -326.7),
    "IB_lsy-mediocre-0.4": (-75.0, -68.9, -90.8, -109., -71.2, -76.8, -80.8, -81.4, -327.4),
    "IB_lsy-mediocre-0.6": (-71.1, -243., -91.3, -111., -78., -74.3, -79.5, -86.1, -327.4),
    "IB_lsy-mediocre-0.8": (-69.7, -62.9, -95.3, -104., -125., -70.7, -79.1, -92.5, -326.9),
    "IB_lsy-mediocre-1.0": (-64.1, -63.8, -113., -65.1, -68.6, -65.4, -66.4, -105.1, -326.8),

    "IB_lsy-optimized-0.0": (-59.8, -60.2, -127., -60.5, -60.1, -60.26, -60.88, -126.2, -283.6),
    "IB_lsy-optimized-0.2": (-60.4, -58.2, -78.4, -61.7, -60.6, -60.61, -60.55, -102.4, -327.4),
    "IB_lsy-optimized-0.4": (-60.8, -58.6, -165., -64.7, -62.4, -61.07, -60.65, -71.97, -327.4),
    "IB_lsy-optimized-0.6": (-62.0, -59.4, -76.9, -64.3, -62.7, -62.68, -60.61, -80.63, -327.4),
    "IB_lsy-optimized-0.8": (-62.7, -61.7, -98.7, -63.1, -74.1, -63.65, -61.29, -90.31, -327.3),
    "IB_lsy-optimized-1.0": (-64.1, -63.8, -113., -65.1, -68.6, -65.4, -66.4, -105.1, -326.8),
    }


if __name__ == "__main__":
    mypfiles = [
        "results/IB_lsy-mediocre-1.0.p",
    ]
    extras = [
        "results/modelfree/IB_lsy-mediocre-1.0.p",
    ]
    plot_performances(mypfiles, "results/ib_global")
    plot_performances(mypfiles, "results/ib_global_modelfree", extra=extras)