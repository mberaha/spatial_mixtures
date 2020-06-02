import argparse
import os
import pickle

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import spatial_mix.utils as spmix_utils

np.random.seed(2129419)


def inv_alr(x):
    out = np.exp(np.hstack((x, 0)))
    return out / np.sum(out)


def get_weights(Nx, Ny):

    N = Nx*Ny
    centers = np.zeros((N, 2))
    for i in range(Nx):
        for j in range(Ny):
            centers[i + j*Nx, :] = np.array([i + 0.5, j + 0.5])
    c = 0.3
    alpha1 = c
    alpha2 = -c
    beta1 = c
    beta2 = -c

    weights = []
    mean_centers = np.mean(centers, axis=0)
    for center in centers:
        w1 = alpha1 * (center[0] - mean_centers[0]) \
             + beta1 * (center[1] - mean_centers[1])
        w2 = alpha2 * (center[0] - mean_centers[0]) \
            + beta2 * (center[1] - mean_centers[1])
        weights.append(inv_alr([w1, w2]))

    return weights


def simulate_from_mixture(weights):
    means = [-5, 0, 5]
    comp = np.random.choice(3, p=weights)
    return np.random.normal(loc=means[comp], scale=1)


def simulate_data(weights, numSamples):
    data = []
    for i in range(len(weights)):
        for j in range(numSamples):
            data.append([i, simulate_from_mixture(weights[i])])
    return pd.DataFrame(data, columns=["group", "datum"])


def compute_G():
    G = np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1) +\
        np.diag(np.ones(N-Nx), Nx) + np.diag(np.ones(N-Nx), -Nx)
    # tolgo i bordi
    border_indices = Nx*np.arange(1, Ny)
    G[border_indices, border_indices - 1] = 0
    G[border_indices - 1, border_indices] = 0

    return G


def run_spmix(data, chain_file, dens_file):
    sp_chains, time = spmix_utils.runSpatialMixtureSampler(
        burnin, niter, thin, W, params_filename, data, [])

    spmix_utils.writeChains(sp_chains, chain_file)
    sp_dens = spmix_utils.estimateDensities(sp_chains, xgrid)

    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": sp_dens}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_data", type=str, default="")
    parser.add_argument("--output_path", type=str, default="data/simulation2/")
    args = parser.parse_args()

    datas = []
    data_filenames = []
    weights_filenames = []
    spmix_chains_filenames = []
    spmix_dens_filenames = []

    nproc = 3

    params_filename = "spatial_mix/resources/sampler_params.asciipb"

    xgrid = np.linspace(-10, 10, 1000)

    data_filenames.append(
        os.path.join(args.output_path +
                     "data.csv"))
    weights_filenames.append(
        os.path.join(args.output_path +
                     "weights.csv"))
    spmix_chains_filenames.append(
        os.path.join(args.output_path +
                     "spmix_chains.recordio"))
    spmix_dens_filenames.append(
        os.path.join(args.output_path +
                     "spmix_dens.pickle"))

    Nx = 4
    Ny = 4
    N = Nx*Ny
    if args.simulate_data:
        weights = get_weights(Nx, Ny)
        datas.append(simulate_data(weights, 100))
        datas[0].to_csv(data_filenames[0], index=False)
        pd.DataFrame(weights).to_csv(weights_filenames[0], index=False)
    else:
        datas.append(pd.read_csv(data_filenames[0]))
        raise Warning('Meglio simulare i dati di nuovo,'
                      + 'non so se è coerente il numero di location con i dati')

    # run models
    burnin = 10000
    niter = 10000
    thin = 5
    ngroups = N
    W = compute_G()

    # first our model, in parallel
    groupedDatas = []

    curr = []
    df = datas[0]
    for g in range(ngroups):
        curr.append(df[df['group'] == g]['datum'].values)
    groupedDatas.append(curr)

    Parallel(n_jobs=nproc)(
        delayed(run_spmix)(data, chain, dens) for data, chain, dens in zip(
            groupedDatas, spmix_chains_filenames, spmix_dens_filenames))
