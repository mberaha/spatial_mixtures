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
import multiprocessing
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
                test_y, test_covs, test_areas):

    chains, time = spmix_utils.runSpatialMixtureSampler(
        5000, 5000, 5, graph, paramsfile,
        resp, covariates, num_comps)

    niter = len(chains)
    preds = np.zeros((niter, len(test_y)))

    numGroups = len(chains[0].groupParams)
    num_components = chains[0].num_components

    means_chain = np.vstack(
        [list(map(lambda x: x.mean, state.atoms)) for state in chains])

    stdevs_chain = np.vstack(
        [list(map(lambda x: x.stdev, state.atoms)) for state in chains])

    regressor_chains = np.vstack(
        [state.regression_coefficients for state in chains])

    weights_chains = []
    for g in range(numGroups):
        weights_chains.append(np.vstack(
            [state.groupParams[g].weights for state in chains]))
    weights_chains = np.stack(weights_chains, axis=-1)

    for i in range(niter):
        means = np.dot(test_covs, regressor_chains[i, :])
        clusters = np.apply_along_axis(
            lambda x: np.random.choice(range(num_comps), p=x), -1, 
            weights_chains[i, :, test_areas.astype(np.int32)])
        errs = norm.rvs(
            loc=means_chain[i, clusters], scale=stdevs_chain[i, clusters])
        preds[i, :] = means + errs

    predmean = np.mean(preds, axis=0)
    return np.sum((predmean - test_y) ** 2) / len(test_y)
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
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--njobs", type=int)

    args = parser.parse_args()


    df = pd.read_csv(args.input_file)
    G = np.loadtxt(args.gmat_file)
    njobs = args.njobs
    num_comps = [5, 10, 15, 20]
    out = {}
    for ncomp in num_comps:
        out[ncomp] = run_cross_val(df, G, ncomp, njobs)


    with open(args.output_file, "wb") as fp:
        pickle.dump(out, fp)











