import argparse
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import pystan

from copy import deepcopy
from joblib import Parallel, delayed
from scipy.stats import t, skewnorm, cauchy, chi2
from subprocess import Popen

import spatial_mix.hdp_utils as hdp_utils
import spatial_mix.utils as spmix_utils


np.random.seed(124151)
xgrid = np.linspace(-10, 10, 1000)


def simulate_data_scenario12(N, M):
    data = []
    for i in range(N):
        data.append([0, t.rvs(6, -4, 1)])

    for i in range(M):
        data.append([1, t.rvs(6, -4, 1)])

    for i in range(N):
        data.append([2, skewnorm.rvs(4, 4, 1)])

    for i in range(M):
        data.append([3, skewnorm.rvs(4, 4, 1)])

    for i in range(N):
        data.append([4, chi2.rvs(3, 0, 1)])

    for i in range(M):
        data.append([5, chi2.rvs(3, 0, 1)])

    return pd.DataFrame(data, columns=["group", "datum"])


def simulate_data_scenario3(N, M):
    data = []
    for i in range(N):
        data.append([0, t.rvs(6, -4, 1)])

    for i in range(M):
        data.append([1, t.rvs(6, -4, 1)])

    for i in range(N):
        data.append([2, skewnorm.rvs(4, 4, 1)])

    for i in range(M):
        data.append([3, skewnorm.rvs(4, 4, 1)])

    for i in range(N):
        data.append([4, cauchy.rvs(0, 1)])

    for i in range(M):
        data.append([5, cauchy.rvs(0, 1)])

    return pd.DataFrame(data, columns=["group", "datum"])


def run_spmix(data, chain_file, dens_file):
    sp_chains = spmix_utils.runSpatialMixtureSampler(
        burnin, niter, thin, W, params_filename, data, [])

    spmix_utils.writeChains(sp_chains, chain_file)
    sp_dens = spmix_utils.estimateDensities(sp_chains, xgrid)

    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": sp_dens}, fp)


def run_hdp(chain_file, data, dens_file):
    hdp_chains = hdp_utils.runHdpSampler(
        burnin, niter, thin, data)

    spmix_utils.writeChains(hdp_chains, chain_file)
    hdp_dens = hdp_utils.estimateDensities(hdp_chains, xgrid)
    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": hdp_dens}, fp)


def run_jo(model, datas, chain_file, dens_file):
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

    fit = model.sampling(data=stan_data, iter=8000, n_jobs=1)
    with open(chain_file, 'wb') as fp:
        pickle.dump({"model": model, "fit": fit}, fp)

    stan_dens = spmix_utils.eval_stan_density(fit, xgrid)
    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": stan_dens}, fp)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", type=str, default="")
    parser.add_argument("--spmix", type=str, default="")
    parser.add_argument("--hdp", type=str, default="")
    parser.add_argument("--jo", type=str, default="")
    parser.add_argument("--output_path", type=str, default="data/simulation1/")
    parser.add_argument("--num_rep", type=int, default=100)
    args = parser.parse_args()

     # run models
    burnin = 10000
    niter = 10000
    thin = 5
    ngroups = 6
    W = np.zeros((ngroups, ngroups))
    W[0, 1] = W[1, 0] = 1
    W[2, 3] = W[3, 2] = 1
    W[4, 5] = W[5, 4] = 1

    params_filename = "spatial_mix/resources/sampler_params.asciipb"

    stan_model = pystan.StanModel(
        file="spatial_mix/resources/mc_car_ssm.stan"
    )

    q = multiprocessing.Queue()
    jobs = []
    for i in range(args.num_rep):
        outdir = os.path.join("rep".format(i))
        os.makedirs(outdir, exist_ok=True)
        if args.load_data:
            datas = []
            for j in range(3):
                datas.append(pd.read_csv(
                    os.path.join(outdir, "data{0}.csv".format(j))))
        else:
            datas = [
                simulate_data_scenario12(1000, 1000),
                simulate_data_scenario12(1000, 10),
                simulate_data_scenario3(100, 100)
            ]

            for j in range(3):
                datas[j].to_csv(os.path.join(outdir, "data{0}.csv".format(j)))

        for j in range(3):
            currdata = []
            df = datas[j]
            for g in range(ngroups):
                currdata.append(df[df['group'] == g]['datum'].values)

            if args.spmix:
                chainfile = os.path.join(
                    outdir, "spmix_chains_scenario{0}.recordio".format(j))
                densfile = os.path.join(
                    outdir, "spmix_dens_scenario{0}.pickle".format(j))

                job1 = multiprocessing.Process(
                    target=run_spmix, args=(currdata, chainfile, densfile))
                job1.start()
                jobs.append(job1)

            if args.hdp:
                chainfile = os.path.join(
                    outdir, "hdp_chains_scenario{0}.recordio".format(j))
                densfile = os.path.join(
                    outdir, "hdp_dens_scenario{0}.pickle".format(j))

                job2 = multiprocessing.Process(
                    target=run_hdp, args=(currdata, chainfile, densfile))
                job2.start()
                jobs.append(job2)

            if args.jo:
                chainfile = os.path.join(
                    outdir, "jo_chains_scenario{0}.recordio".format(j))
                densfile = os.path.join(
                    outdir, "jo_dens_scenario{0}.pickle".format(j))

                job3 = multiprocessing.Process(
                    target=run_jo, args=(
                        deepcopy(stan_model), currdata, chainfile, densfile))
                job3.start()
                jobs.append(job3)

    for j in jobs:
        j.join()
