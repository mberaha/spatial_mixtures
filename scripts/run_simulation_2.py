# run from spatial_lda: python3 -m scripts.run_simulation2

import argparse
import os
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import norm
import time

from scipy.integrate import simps


import spatial_mix.utils as spmix_utils
import spatial_mix.hdp_utils as hdp_utils


np.random.seed(2129419)
xgrid = np.linspace(-10, 10, 1000)
<<<<<<< HEAD
=======


def hellinger_dist(p, q, xgrid):
    return np.sqrt(0.5 * simps((np.sqrt(p) - np.sqrt(q)) ** 2, xgrid))


def post_hellinger_dist(estimatedDens, true, xgrid):
    return np.apply_along_axis(
        lambda x: hellinger_dist(x, true, xgrid), 1, estimatedDens)


def kl_div(p, q, xgrid):
    return simps(p * (np.log(p + 1e-5) - np.log(q + 1e-5)), xgrid)


def post_kl_div(estimatedDens, true, xgrid):
    return np.apply_along_axis(
        lambda x: kl_div(true, x, xgrid), 1, estimatedDens)
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3


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


def true_densities(xgrid, weights):
    means = [-5, 0, 5]
    true_dens = []
    for w in weights:
        true_dens.append(w[0] * norm.pdf(xgrid, means[0], 1.0) +
                         w[1] * norm.pdf(xgrid, means[1], 1.0) +
                         w[2] * norm.pdf(xgrid, means[2], 1.0))
    return true_dens


def simulate_data(weights, numSamples):
    data = []
    for i in range(len(weights)):
        for j in range(numSamples):
            data.append([i, simulate_from_mixture(weights[i])])
    return pd.DataFrame(data, columns=["group", "datum"])


def compute_G(Nx, Ny):
    N = Nx*Ny
    G = np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1) +\
        np.diag(np.ones(N-Nx), Nx) + np.diag(np.ones(N-Nx), -Nx)
    # tolgo i bordi
    border_indices = Nx*np.arange(1, Ny)
    G[border_indices, border_indices - 1] = 0
    G[border_indices - 1, border_indices] = 0

    return G


def run_spmix(data, dens_file, true_dens, index, rep, times):
    sp_chains, time = spmix_utils.runSpatialMixtureSampler(
        burnin, niter, thin, W, params_filename, data, [])

    times[index, rep] = time

    # spmix_utils.writeChains(sp_chains, chain_file)
    sp_dens = spmix_utils.estimateDensities(sp_chains, xgrid)
    save_errors(sp_dens, true_dens, rep, dens_file)


def run_hdp(data, dens_file, true_dens, index, rep, times):
    hdp_chains, time = hdp_utils.runHdpSampler(
        burnin, niter, thin, data)

    times[index, rep] = time
    # spmix_utils.writeChains(hdp_chains, chain_file)
    hdp_dens = hdp_utils.estimateDensities(hdp_chains, xgrid)

    save_errors(hdp_dens, true_dens, rep, dens_file)

    # with open(dens_file, "wb") as fp:
    #     pickle.dump({"xgrid": xgrid, "dens": hdp_dens}, fp)


def save_errors(estimate_dens, true_dens, rep, dens_file):

    kl_divs = []
    hell_dists = []
    for i, dens in enumerate(estimate_dens):

        hell_dists.append((rep, i, np.mean(post_hellinger_dist(
            dens, true_dens[i], xgrid))))

        kl_divs.append((rep, i, np.mean(post_kl_div(
            dens, true_dens[i], xgrid))))

    out = {'xgrid': xgrid, 'hell_dist': hell_dists, 'kl_divs': kl_divs}

    with open(dens_file, "wb") as fp:
        pickle.dump(out, fp)


def run_hdp(data, chain_file, dens_file):
    hdp_chains = hdp_utils.runHdpSampler(
        burnin, niter, thin, data)

    spmix_utils.writeChains(hdp_chains, chain_file)
    hdp_dens = hdp_utils.estimateDensities(hdp_chains, xgrid)
    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": hdp_dens}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="data/simulation2/")
    parser.add_argument("--njobs", type=int, default=4)
    args = parser.parse_args()

<<<<<<< HEAD
    nproc = 4

=======
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
    outdir_sp = os.path.join(args.output_path, "spmix")
    os.makedirs(outdir_sp, exist_ok=True)

    outdir_hdp = os.path.join(args.output_path, "hdp")
    os.makedirs(outdir_hdp, exist_ok=True)

    params_filename = "spatial_mix/resources/sampler_params.asciipb"

<<<<<<< HEAD
    Nx = [2, 4, 8, 16]
=======
    Nx = [2, 4, 8, 16, 32]
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
    num_repetions = 10
    num_data_per_group = 50

    burnin = 10000
    niter = 10000
    thin = 5

    q = multiprocessing.Queue()
    jobs = []

    curr_jobs = 0

<<<<<<< HEAD
    # number of locations
    for n in Nx:
=======
    sp_times = np.zeros((len(Nx), num_repetions))
    hdp_times = np.zeros((len(Nx), num_repetions))

    # number of locations
    for index, n in enumerate(Nx):
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
        ngroups = n**2
        W = compute_G(n, n)

        # create
<<<<<<< HEAD
        chaindir_sp = os.path.join(outdir_sp, "chains/areas{0}".format(ngroups))
        densdir_sp = os.path.join(outdir_sp, "dens/areas{0}".format(ngroups))
        chaindir_hdp = os.path.join(outdir_hdp, "chains/areas{0}".format(ngroups))
        densdir_hdp = os.path.join(outdir_hdp, "dens/areas{0}".format(ngroups))

        os.makedirs(chaindir_sp, exist_ok=True)
        os.makedirs(densdir_sp, exist_ok=True)
        os.makedirs(chaindir_hdp, exist_ok=True)
=======
        densdir_sp = os.path.join(outdir_sp, "dens/areas{0}".format(ngroups))
        densdir_hdp = os.path.join(outdir_hdp, "dens/areas{0}".format(ngroups))

        os.makedirs(densdir_sp, exist_ok=True)
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
        os.makedirs(densdir_hdp, exist_ok=True)

        # repetitions
        for rep in range(num_repetions):
            # simulate data
            weights = get_weights(n, n)
            datas = simulate_data(weights, num_data_per_group)
<<<<<<< HEAD

=======
            true_dens = true_densities(xgrid, weights)
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
            # first our model, in parallel
            groupedData = []
            for g in range(ngroups):
                groupedData.append(datas[datas['group'] == g]['datum'].values)

            # spmix
<<<<<<< HEAD
            chainfile = os.path.join(chaindir_sp, "{0}.recordio".format(rep))
            densfile = os.path.join(densdir_sp, "{0}.pickle".format(rep))

            job1 = multiprocessing.Process(
                target=run_spmix, args=(groupedData, chainfile, densfile))
=======
            densfile = os.path.join(densdir_sp, "{0}.pickle".format(rep))
            start_sp = time.time()
            job1 = multiprocessing.Process(
                target=run_spmix, args=(groupedData, densfile, true_dens, index,
                rep, sp_times))
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
            job1.start()
            jobs.append(job1)
            curr_jobs += 1

            # hdp
<<<<<<< HEAD
            chainfile = os.path.join(
                chaindir_hdp, "{0}.recordio".format(rep))
            densfile = os.path.join(densdir_hdp, "{0}.pickle".format(rep))

            job2 = multiprocessing.Process(
                target=run_hdp, args=(groupedData, chainfile, densfile))
=======
            densfile = os.path.join(densdir_hdp, "{0}.pickle".format(rep))

            job2 = multiprocessing.Process(
                target=run_hdp, args=(groupedData, densfile, true_dens, index,
                rep, hdp_times))
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
            job2.start()
            jobs.append(job2)
            curr_jobs += 1

            if curr_jobs == args.njobs:
                for j in jobs:
                    j.join()

            jobs = []
            curr_jobs = 0

        for j in jobs:
            j.join()
<<<<<<< HEAD
=======

    # save times
    with open(os.path.join(args.output_path, "times.pickle"), "wb") as fp:
        pickle.dump({"sp_times": sp_times, "hdp_times": hdp_times}, fp)
>>>>>>> d5a91d45a93288a65d878b3c014c569f031d2fb3
