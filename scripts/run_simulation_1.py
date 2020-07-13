import argparse
import glob
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


def run_spmix(data, chain_file, dens_file):
    sp_chains, time = spmix_utils.runSpatialMixtureSampler(
        burnin, niter, thin, W, params_filename, data, [])

    spmix_utils.writeChains(sp_chains, chain_file)
    sp_dens = spmix_utils.estimateDensities(sp_chains, xgrid)

    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": sp_dens}, fp)


def run_hdp(data, chain_file, dens_file):
    hdp_chains, time = hdp_utils.runHdpSampler(
        burnin, niter, thin, data)

    spmix_utils.writeChains(hdp_chains, chain_file)
    hdp_dens = hdp_utils.estimateDensities(hdp_chains, xgrid)
    with open(dens_file, "wb") as fp:
        pickle.dump({"xgrid": xgrid, "dens": hdp_dens}, fp)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--spmix", type=str, default="")
    parser.add_argument("--hdp", type=str, default="")
    parser.add_argument("--output_path", type=str, default="data/simulation1/")
    parser.add_argument("--num_rep", type=int, default=100)
    parser.add_argument("--njobs", type=int, default=6)

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

    outdir_sp = os.path.join(args.output_path, "spmix")
    os.makedirs(outdir_sp, exist_ok=True)
    

    outdir_hdp = os.path.join(args.output_path, "hdp")
    os.makedirs(outdir_hdp, exist_ok=True)
 

    q = multiprocessing.Queue()
    jobs = []

    curr_jobs = 0

    for j in range(3):
        filenames = glob.glob(os.path.join(
            args.data_path, "scenario{0}/*".format(j)))

        chaindir_sp = os.path.join(outdir_sp, "chains/scenario{0}".format(j))
        densdir_sp = os.path.join(outdir_sp, "dens/scenario{0}".format(j))
        chaindir_hdp = os.path.join(outdir_hdp, "chains/scenario{0}".format(j))
        densdir_hdp = os.path.join(outdir_hdp, "dens/scenario{0}".format(j))

        os.makedirs(chaindir_sp, exist_ok=True)
        os.makedirs(densdir_sp, exist_ok=True)
        os.makedirs(chaindir_hdp, exist_ok=True)
        os.makedirs(densdir_hdp, exist_ok=True)

        for filename in filenames:
            rep = filename.split("/")[-1].split(".")[0]
            df = pd.read_csv(filename)

            currdata = []
            for g in range(ngroups):
                currdata.append(df[df['group'] == g]['datum'].values)

            if args.spmix:
                chainfile = os.path.join(chaindir_sp, "{0}.recordio".format(rep))
                densfile = os.path.join(densdir_sp, "{0}.pickle".format(rep))

                job1 = multiprocessing.Process(
                    target=run_spmix, args=(currdata, chainfile, densfile))
                job1.start()
                jobs.append(job1)
                curr_jobs += 1
                

            if args.hdp:
                chainfile = os.path.join(
                    chaindir_hdp, "{0}.recordio".format(rep))
                densfile = os.path.join(densdir_hdp, "{0}.pickle".format(rep))

                job2 = multiprocessing.Process(
                    target=run_hdp, args=(currdata, chainfile, densfile))
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
