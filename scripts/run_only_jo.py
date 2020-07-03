"""
This scripts runs the CK-CAR model from Jo el al (2017) for the simulated
example #1
To generate data, use the script scripts/generate_simulation1_data.py

Usage: from the root folder (../), run
python3 -m scripts.run_only_jo \
    --data_path=<path_to_data_files> --output_path=<path_to_output_folde>
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


xgrid = np.linspace(-10, 10, 1000)

true_dens_scenario12 = [
    t.pdf(xgrid, 6, -4, 1), t.pdf(xgrid, 6, -4, 1),
    skewnorm.pdf(xgrid, 4, 4, 1), skewnorm.pdf(xgrid, 4, 4, 1),
    chi2.pdf(xgrid, 3, 0, 1), chi2.pdf(xgrid, 3, 0, 1)
]

true_dens_scenario3 = [
    t.pdf(xgrid, 6, -4, 1), t.pdf(xgrid, 6, -4, 1),
    skewnorm.pdf(xgrid, 4, 4, 1), skewnorm.pdf(xgrid, 4, 4, 1),
    cauchy.pdf(xgrid, 0, 1), cauchy.pdf(xgrid, 0, 1)
]


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


def run_jo(
        model, datas, chain_file, dens_file,
        scen, save_chain, save_dens):
    print("************** STARTING {0} ***************".format(chain_file))

    data_by_group_stan = []
    max_num_data = np.max([len(x) for x in datas])
    for i in range(6):
        data_by_group_stan.append(
            np.concatenate([datas[i], np.zeros(max_num_data - len(datas[i]))]))

    stan_data = {
        "num_groups": 6,
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
    for loc in range(6):
        true_d = true_dens_scenario12[loc] if scen < 2 else true_dens_scenario3[loc]
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
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--njobs", type=int, default=6)

    args = parser.parse_args()

    outdir = os.path.join(args.output_path, "jo")
    os.makedirs(outdir, exist_ok=True)

    
    ngroups = 6
    W = np.zeros((ngroups, ngroups))
    W[0, 1] = W[1, 0] = 1
    W[2, 3] = W[3, 2] = 1
    W[4, 5] = W[5, 4] = 1

    stan_model = pystan.StanModel(
        file="spatial_mix/resources/mc_car_ssm.stan")

    q = multiprocessing.Queue()
    jobs = []
    curr_jobs = 0
    for scen in list(range(3)):
        filenames = glob.glob(os.path.join(
            args.data_path, "scenario{0}/*".format(scen)))

        chaindir = os.path.join(outdir, "chains/scenario{0}".format(scen))
        densdir = os.path.join(outdir, "dens/scenario{0}".format(scen))
        os.makedirs(chaindir, exist_ok=True)
        os.makedirs(densdir, exist_ok=True)

        for filename in filenames:
            rep = filename.split("/")[-1].split(".")[0]
            if rep == 1:
                save_chain = True
                save_dens = True
            else:
                save_chain = False
                save_dens = False

            chainfile = os.path.join(chaindir, "{0}.pickle".format(rep))
            densfile = os.path.join(densdir, "{0}.pickle".format(rep))
            df = pd.read_csv(filename)
            currdata = []
            for g in range(ngroups):
                currdata.append(df[df['group'] == g]['datum'].values)

            job = multiprocessing.Process(
                target=run_jo, args=(
                    deepcopy(stan_model), currdata, chainfile, densfile,
                    scen, save_chain, save_dens))
            job.start()
            jobs.append(job)
            curr_jobs += 1

            if curr_jobs == args.njobs:
                for j in jobs:
                    j.join()

                jobs = []
                curr_jobs = 0

    for j in jobs:
        j.join()
