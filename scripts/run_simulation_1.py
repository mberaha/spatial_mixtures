import argparse
import os
import pickle

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from scipy.stats import t, skewnorm, cauchy, chi2
from subprocess import Popen

import spatial_mix.hdp_utils as hdp_utils
import spatial_mix.utils as spmix_utils
from spatial_mix.protos.py.univariate_mixture_state_pb2 import HdpState

np.random.seed(21294192849184)


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


def analyze_hdp(chain_file, dens_file):
    hdp_chains = spmix_utils.loadChains(chain_file, HdpState)
    hdp_dens = hdp_utils.estimateDensities(hdp_chains, xgrid)
    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": hdp_dens}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_data", type=str, default="")
    parser.add_argument("--output_path", type=str, default="data/simulation1/")
    args = parser.parse_args()

    datas = []
    data_filenames = []
    spmix_chains_filenames = []
    spmix_dens_filenames = []
    hdp_chains_filenames = []
    hdp_dens_filenames = []
    nproc = 3

    params_filename = "spatial_mix/resources/sampler_params.asciipb"

    xgrid = np.linspace(-10, 10, 1000)
    for i in range(3):
        data_filenames.append(
            os.path.join(args.output_path +
                         "data_scenario{0}.csv".format(i+1)))
        spmix_chains_filenames.append(
            os.path.join(args.output_path +
                         "spmix_chains_scenario{0}.recordio".format(i+1)))
        spmix_dens_filenames.append(
            os.path.join(args.output_path +
                         "spmix_dens_scenario{0}.pickle".format(i+1)))

        hdp_chains_filenames.append(
            os.path.join(args.output_path +
                         "hdp_chains_scenario{0}.recordio".format(i+1)))
        hdp_dens_filenames.append(
            os.path.join(args.output_path +
                         "hdp_dens_scenario{0}.pickle".format(i+1)))

    if args.simulate_data:
        datas.append(simulate_data_scenario12(1000, 1000))
        datas.append(simulate_data_scenario12(1000, 10))
        datas.append(simulate_data_scenario3(100, 100))

        for i in range(3):
            datas[i].to_csv(data_filenames[i], index=False)

    else:
        for i in range(3):
            datas.append(pd.read_csv(data_filenames[i]))

    # run models
    burnin = 10000
    niter = 10000
    thin = 5
    ngroups = 6
    W = np.zeros((ngroups, ngroups))
    W[0, 1] = W[1, 0] = 1
    W[2, 3] = W[3, 2] = 1
    W[4, 5] = W[5, 4] = 1

    # first our model, in parallel
    groupedDatas = []

    for i in range(3):
        curr = []
        df = datas[i]
        for g in range(ngroups):
            curr.append(df[df['group'] == g]['datum'].values)
        groupedDatas.append(curr)

    Parallel(n_jobs=nproc)(
        delayed(run_spmix)(data, chain, dens) for data, chain, dens in zip(
            groupedDatas, spmix_chains_filenames, spmix_dens_filenames))

    commands = []
    for i in range(3):
        commands.append(" ".join([
            "./spatial_mix/run_hdp_from_file.out",
            data_filenames[i], hdp_chains_filenames[i]]).split())

    procs = [Popen(i) for i in commands]

    for p in procs:
        p.wait()

    for i in range(3):
        analyze_hdp(hdp_chains_filenames[i], hdp_dens_filenames[i])
