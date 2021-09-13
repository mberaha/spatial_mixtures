"""
This scripts runs the CK-CAR model from Jo el al (2017) for the simulated
example #2

Usage: from the root folder (../), run
python3 -m scripts.run_simu2_jo --output_path=<path_to_output_folde>
"""

import argparse
import glob
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import pystan

from copy import deepcopy
from scipy.stats import t, norm, skewnorm, cauchy, chi2
from scipy.integrate import simps, trapz



# true_dens_scenario12 = [
#     t.pdf(xgrid, 6, -4, 1), t.pdf(xgrid, 6, -4, 1),
#     skewnorm.pdf(xgrid, 4, 4, 1), skewnorm.pdf(xgrid, 4, 4, 1),
#     chi2.pdf(xgrid, 3, 0, 1), chi2.pdf(xgrid, 3, 0, 1)
# ]

# true_dens_scenario3 = [
#     t.pdf(xgrid, 6, -4, 1), t.pdf(xgrid, 6, -4, 1),
#     skewnorm.pdf(xgrid, 4, 4, 1), skewnorm.pdf(xgrid, 4, 4, 1),
#     cauchy.pdf(xgrid, 0, 1), cauchy.pdf(xgrid, 0, 1)
# ]

np.random.seed(2129419)
xgrid = np.linspace(-10, 10, 1000)


def inv_alr(x):
    out = np.exp(np.hstack((x, 0)))
    return out / np.sum(out)


def eval_density(weights, means, stdevs, xgrid):
    """
    Estimate the density of a normal mixture with parameters
    'weights', 'means' and 'stdevs' over 'xgrid'
    """
    return np.dot(norm.pdf(
        np.hstack([xgrid.reshape(-1, 1)] * len(means)), means, stdevs),
        weights)


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


def compute_G(Nx, Ny):
    N = Nx*Ny
    G = np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1) +\
        np.diag(np.ones(N-Nx), Nx) + np.diag(np.ones(N-Nx), -Nx)
    # tolgo i bordi
    border_indices = Nx*np.arange(1, Ny)
    G[border_indices, border_indices - 1] = 0
    G[border_indices - 1, border_indices] = 0

    return G


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


def eval_stan_density(stanfit, xgrid):
    means = stanfit.extract("means")["means"]
    variances = stanfit.extract("vars")["vars"]
    weights = stanfit.extract("weights")["weights"]
    out = []
    num_iters = means.shape[0]
    num_components = means.shape[1]

    means = means.reshape(-1)
    stddevs = np.sqrt(variances.reshape(-1))
    allgrid = np.hstack([xgrid.reshape(-1, 1)] * means.shape[0])

    eval_normals = norm.pdf(
        allgrid, means, stddevs
    ).reshape(len(xgrid), num_iters, num_components)

    numGroups = weights.shape[1]
    for g in range(numGroups):
        weights_chain = weights[:, g, :]
        out.append(np.sum(eval_normals*weights_chain, axis=-1).T)

    return out


def run_jo(model, datas, chain_file, dens_file,
           W, scen, true_dens, save_chain, save_dens):
    print("************** STARTING {0} ***************".format(chain_file))

    data_by_group_stan = []
    max_num_data = np.max([len(x) for x in datas])
    for i in range(len(datas)):
        data_by_group_stan.append(
            np.concatenate([datas[i], np.zeros(max_num_data - len(datas[i]))]))

    stan_data = {
        "num_groups": len(datas),
        "num_data_per_group": [len(x) for x in datas],
        "max_data_per_group": np.max([len(x) for x in datas]),
        "num_components": 10,
        "data_by_group": data_by_group_stan,
        "G": W,
        "rho": 0.95,
        "a": 0.1,
        "b": 0.5,
        "points_in_grid": len(xgrid),
        "xgrid": xgrid}

    fit = model.sampling(data=stan_data, iter=8000, n_jobs=1, chains=1)
    if save_chain:
        with open(chain_file, 'wb') as fp:
            pickle.dump({"model": model, "fit": fit}, fp)

    stan_dens = eval_stan_density(fit, xgrid)
    kl_divs = []
    hell_dists = []
    for loc in range(len(datas)):
        true_d = true_dens[loc]
        hell_dists.append((scen, rep, loc, np.mean(post_hellinger_dist(
            stan_dens[loc], true_d, xgrid))))

        kl_divs.append((scen, rep, loc, np.mean(post_kl_div(
            stan_dens[loc], true_d, xgrid))))

    out = {'xgrid': xgrid, 'hell_dits': hell_dists, 'kl_divs': kl_divs}
    if save_dens:
        out["dens"] = stan_dens

    with open(dens_file, "wb") as fp:
        pickle.dump(out, fp)

    print("************** FINISHED {0} ***************".format(chain_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--njobs", type=int, default=6)

    args = parser.parse_args()

    outdir = os.path.join(args.output_path, "jo")
    os.makedirs(outdir, exist_ok=True)

    stan_model = pystan.StanModel(
        file="spatial_mix/resources/mc_car_ssm.stan")

    Nx = [2, 4, 8, 16]
    num_repetions = 10
    num_data_per_group = 25

    q = multiprocessing.Queue()
    jobs = []
    curr_jobs = 0

    for n in Nx:
        ngroups = n**2
        W = compute_G(n, n)

        chaindir = os.path.join(outdir, "chains/scenario{0}".format(ngroups))
        densdir = os.path.join(outdir, "dens/scenario{0}".format(ngroups))
        os.makedirs(chaindir, exist_ok=True)
        os.makedirs(densdir, exist_ok=True)

        for rep in range(num_repetions):
            # simulate data
            weights = get_weights(n, n)
            datas = simulate_data(weights, num_data_per_group)
            true_dens = [
                eval_density(weights[j], np.array([-5, 0, 5]), np.ones(3), xgrid) for j in ngroups]

            # first our model, in parallel
            groupedData = []
            for g in range(ngroups):
                groupedData.append(datas[datas['group'] == g]['datum'].values)

            if rep == 1:
                save_chain = True
                save_dens = True
            else:
                save_chain = False
                save_dens = False

            # spmix
            # chainfile = os.path.join(chaindir_sp, "{0}.recordio".format(rep))
            # densfile = os.path.join(densdir_sp, "{0}.pickle".format(rep))

            chainfile = os.path.join(chaindir, "{0}.pickle".format(rep))
            densfile = os.path.join(densdir, "{0}.pickle".format(rep))

            job1 = multiprocessing.Process(
                target=run_jo, args=(
                    deepcopy(stan_model), groupedData, chainfile, densfile,
                    W, ngroups, true_dens, save_chain, save_dens))
            job1.start()
            jobs.append(job1)
            curr_jobs += 1

            if curr_jobs == args.njobs:
                for j in jobs:
                    j.join()

                jobs = []
                curr_jobs = 0

        for j in jobs:
            j.join()
