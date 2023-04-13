"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from generate_data import go_either
from simpleenv import gaussian_pdf

def flatten(dataset):
    return [tuple_obj for sublist in dataset for tuple_obj in sublist]

def plot_heatmap(dataset, policy, analysis_path):
    plt.rcParams.update({'font.size': 14})
    dataset = flatten(dataset)
    states = np.vstack([x[0] for x in dataset])

    # pic 1

    fig, axes = plt.subplots(1,4, figsize=(20,5))
    axes[0].set_title("Scatter of visited states")
    axes[0].scatter(states[:, 0], states[:, 1])
    axlim = 10
    axes[0].set_xlim((0, axlim))
    axes[0].set_ylim((0, axlim))

    # pic 2

    h, xe, ye = np.histogram2d(states[:, 0], states[:, 1], bins=10, range=[[0., axlim], [0., axlim]])
    h_norm = h / np.sum(h)
    h_norm = h_norm.T
    h_norm = np.flip(h_norm, 0)

    axes[1].set_title(f"Heatmap of visited states")
    second_max = sorted(h_norm.flatten())[-2]
    im = axes[1].imshow(h_norm, cmap="plasma", vmin=0, vmax=second_max, extent=[0,axlim,0,axlim])
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='p(s)')

    # pic 3

    states_grid = np.vstack([[x,y] for x in np.arange(0, axlim + 0.5, 0.5) for y in np.arange(0, axlim + 0.5, 0.5)])
    actions_grid = policy(states_grid)
    action_norms = np.linalg.norm(actions_grid, axis=1).reshape((-1, 1))
    actions_normed = actions_grid / action_norms

    angles = np.arccos(actions_normed[:,0])  # those are the angles to (1,0) - no matter which direction
    angles[actions_normed[:,1] < 0] = math.pi + (math.pi - angles[actions_normed[:,1] < 0])  # adjusted for direction
    root_dim = int(np.sqrt(angles.shape[0]))
    angles = angles.reshape((root_dim, root_dim))
    angles = angles.T
    angles = np.flip(angles, 0)

    axes[2].set_title("(a) Behavioral Policy w/o noise")
    im = axes[2].imshow(angles, cmap="jet", vmin=0, vmax=2*math.pi, extent=[0,axlim,0,axlim])
    axes[2].quiver(states_grid[:, 0], states_grid[:, 1], actions_normed[:, 0], actions_normed[:, 1])
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, label='angle to (1,0)')
    cbar.ax.get_yaxis().set_ticks([0., 0.5*math.pi, 1.0*math.pi, 1.5*math.pi, 2.0*math.pi])
    cax.set_yticklabels(["$0$", "$0.5\pi$", "$1.0\pi$", "$1.5\pi$", "$2.0\pi$"])

    # pic 4

    states_grid = np.vstack([[x,y] for x in np.arange(0, axlim + 0.01, 0.01) for y in np.arange(0, axlim + 0.01, 0.01)])
    max_reward = gaussian_pdf(np.array([[3., 6.]]), np.array([[3., 6.]]), 1.5)[0, 0]
    rewards_grid = gaussian_pdf(states_grid, np.array([[3., 6.]]), 1.5)
    rewards_grid = rewards_grid / max_reward

    root_dim = int(np.sqrt(rewards_grid.shape[0]))
    rewards = rewards_grid.reshape((root_dim, root_dim))
    rewards = rewards.T
    rewards = np.flip(rewards, 0)

    axes[3].set_title("(b) 2D Env reward distribution")
    im = axes[3].imshow(rewards, cmap="jet", vmin=0, vmax=1., extent=[0,axlim,0,axlim])
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, label='reward')
    cbar.ax.get_yaxis().set_ticks([0., 0.5, 1.])
    print(max_reward)
    cax.set_yticklabels(["$0$", "$0.5$", "$1.0$"])

    fig.tight_layout()
    plt.savefig(analysis_path, bbox_inches='tight')


if __name__ == "__main__":
    mysavepath = "datasets/either_0.1"
    data = pickle.load(open(mysavepath, "rb"))
    analyze_path = "results/either_0.1_dataset_analysis.png"
    plot_heatmap(data, go_either, analyze_path)