"""
This scripts runs the cross-validation on the Amsterdam dataset to 
perform sensitivity analysis on the choice of the number of components H.

Usage: from the root folder (../), run
python3 -m scripts.run_amsterdam \
    --input_file=<path_to_amsterdam_dataset (csv format)> \
    --gmat_file=<path_to_matrix_G_file (npy format)> \
    --output_file=<output_file>
"""

import argparse
import json
import multiprocessing
import os
import pickle
import numpy as np
import pandas as pd

from scipy.stats import norm

from google.protobuf import text_format
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.model_selection import StratifiedKFold
from spatial_mix.protos.py.sampler_params_pb2 import SamplerParams

import spatial_mix.utils as spmix_utils


paramsfile = "spatial_mix/resources/sampler_params.asciipb"


def run_sampler(resp, covariates, graph, num_comps,
                base_outfile, mu0=None, a=None, b=None, lam=None):

    other_params = {
        "num_comps": num_comps, "mu0": mu0, "a": a, "b": b, "lam": lam}
    other_params = {k: v for k, v in other_params.items() if v is not None}
    print("running with params")
    print(json.dumps(other_params))

    chains, time = spmix_utils.runSpatialMixtureSampler(
        5000, 5000, 5, graph, paramsfile,
        resp, covariates, other_params)

    outfile = base_outfile.format(mu0, a, b, lam)
    spmix_utils.writeChains(chains, outfile)
    return 0


def groupbyneighbor(df):
    areas = np.unique(df.group.values)
    responses = []
    covariates = []
    covnames = df.columns[3:]
    for area in areas:
        currdf = df.loc[df.group == area]
        responses.append(currdf.response.values)
        covariates.append(currdf[covnames].values)
    return responses, covariates


def _run_wrapper(data, G, num_comps,
                 train_index, test_index):
    traindata = data.loc[train_index]
    testdata = data.loc[test_index]

    resp, covs = groupbyneighbor(traindata)
    return run_sampler(
        resp, covs, G, num_comps,
        testdata.response, testdata[df.columns[3:]], testdata.group)


def run_cross_val(data, G, num_comps, njobs):
    ndata = len(data)
    kf = StratifiedKFold(n_splits=10)

    fd = delayed(_run_wrapper)
    out = []
    out = Parallel(n_jobs=njobs)(
        fd(data, G, num_comps, *s)
        for s in kf.split(data, data.group))
    # for train_ind, test_ind in kf.split(data, data.group):
    #     out.append(_run_wrapper(data, G, num_comps,
    #                             train_ind, test_ind))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--gmat_file", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--njobs", type=int)

    args = parser.parse_args()

    base_outfile = os.path.join(
        args.output_folder, "chains_mu0_{0}_a_{1}_b_{2}_lam{3}.recordio")

    df = pd.read_csv(args.input_file)
    G = np.loadtxt(args.gmat_file)
    njobs = args.njobs

    resp, covs = groupbyneighbor(df)

    num_comps = [5]
    mu0s = [0.0]
    a_s = [0.5, 1.0, 2.0, 3.0]
    b_s = [0.5, 1.0, 2.0, 3.0]
    lambdas = [0.05, 0.1, 0.5, 1.0, 2.0]

    q = multiprocessing.Queue()
    jobs = []
    curr_jobs = 0
    for a in a_s:
        for b in b_s:
            for lam in lambdas:
                job = multiprocessing.Process(
                    target=run_sampler, args=(
                        resp, covs, G, num_comps[0], base_outfile,
                        mu0s[0], a, b, lam))
                jobs.append(job)
                curr_jobs += 1

                if curr_jobs == args.njobs:
                    for j in jobs:
                        j.start()

                    for j in jobs:
                        j.join()

                    jobs = []
                    curr_jobs = 0

    for j in jobs:
        j.join()

    # for ncomp in num_comps:
    #     out[ncomp] = run_cross_val(df, G, ncomp, njobs)

    # with open(args.output_file, "wb") as fp:
    #     pickle.dump(out, fp)
