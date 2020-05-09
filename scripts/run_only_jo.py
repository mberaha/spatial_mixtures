import argparse
import glob
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import pystan

from copy import deepcopy
from scipy.stats import norm


xgrid = np.linspace(-10, 10, 1000)


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


def run_jo(model, datas, chain_file, dens_file):
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
    with open(chain_file, 'wb') as fp:
        pickle.dump({"model": model, "fit": fit}, fp)

    stan_dens = eval_stan_density(fit, xgrid)
    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": stan_dens}, fp)

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
    for j in list(range(3)):
        filenames = glob.glob(os.path.join(
            args.data_path, "scenario{0}/*".format(j)))

        chaindir = os.path.join(outdir, "chains/scenario{0}".format(j))
        densdir = os.path.join(outdir, "dens/scenario{0}".format(j))
        os.makedirs(chaindir, exist_ok=True)
        os.makedirs(chaindir, exist_ok=True)

        for filename in filenames:
            rep = filename.split("/")[-1].split(".")[0]
            chainfile = os.path.join(chaindir, "{0}.pickle".format(rep))
            densfile = os.path.join(densdir, "{0}.pickle".format(rep))
            print(chainfile)

            df = pd.read_csv(filename)

            currdata = []
            for g in range(ngroups):
                currdata.append(df[df['group'] == g]['datum'].values)

            job = multiprocessing.Process(
                target=run_jo, args=(
                    deepcopy(stan_model), currdata, chainfile, densfile))
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
