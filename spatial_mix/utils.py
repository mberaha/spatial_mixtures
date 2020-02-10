import logging
import numpy as np
import multiprocessing
import os
import sys

from google.protobuf import text_format
from google.protobuf.internal.decoder import _DecodeVarint32
from scipy.stats import norm
from scipy.integrate import simps

from spatial_mix.protos.py.sampler_params_pb2 import SamplerParams
from spatial_mix.protos.py.univariate_mixture_state_pb2 import UnivariateState

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import spmixtures


def loadChains(filename, msgType=UnivariateState):
    out = []
    with open(filename, "rb") as fp:
        buf = fp.read()

    n = 0
    while n < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, n)
        n = new_pos
        msg_buf = buf[n:n+msg_len]
        try:
            msg = msgType()
            msg.ParseFromString(msg_buf)
            out.append(msg)
            n += msg_len
        except Exception as e:
            break

    return out


def estimateDensity(weights, means, stdevs, xgrid):
    return np.dot(norm.pdf(
        np.hstack([xgrid.reshape(-1, 1)] * len(means)), means, stdevs),
                  weights)


def _aux(state, g, xgrid):
    return estimateDensity(
        state.groupParams[g].weights, state.atoms, xgrid)


def estimateDensities(chains, xgrids, nproc=-1):
    numGroups = len(chains[0].groupParams)
    if not isinstance(xgrids, list):
        xgrids = [xgrids] * numGroups
    numIters = len(chains)
    if nproc == -1:
        nproc = multiprocessing.cpu_count() - 1

    out = []
    num_components = chains[0].num_components
    means_chain = np.vstack(
        [list(map(lambda x: x.mean, state.atoms)) for state in chains]
        ).reshape(-1)
    stdevs_chain = np.vstack(
        [list(map(lambda x: x.stdev, state.atoms)) for state in chains]
        ).reshape(-1)

    for g in range(numGroups):
        weights_chain = np.vstack(
            [state.groupParams[g].weights for state in chains])

        eval_normals = norm.pdf(np.hstack([xgrids[g].reshape(-1, 1)] * means_chain.shape[0]),
                                means_chain,
                                stdevs_chain
                                ).reshape(len(xgrids[g]),
                                          numIters,
                                          num_components)
        out.append(np.sum(eval_normals*weights_chain, axis=-1).T)
    return out


def lpml(densities):
    if isinstance(densities, list):
        densities = np.hstack(densities)
    return np.sum(1 / np.mean(1 / densities, axis=0))


def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out


def hellinger_dist(p, q, xgrid):
    return 0.5 * simps(np.sqrt(p) - np.sqrt(q), xgrid) ** 2


def post_hellinger_dist(estimatedDens, true, xgrid):
    return np.apply_along_axis(
        lambda x: hellinger_dist(x, true, xgrid), 1, estimatedDens)


def kl_div(p, q, xgrid):
    return simps(p * np.log(p + 1e-5) - np.log(q + 1e-5), xgrid)


def post_kl_div(estimatedDens, true, xgrid):
    return np.apply_along_axis(
        lambda x: kl_div(x, true, xgrid), 1, estimatedDens)


def runSpatialMixtureSampler(
        burnin, niter, thin, W, params, data, covariates=[]):

    def checkFromFiles(data, W):
        return isinstance(data, str) and isinstance(W, str)

    def checkFromData(data, W):
        return isinstance(data, list) and \
                all(isinstance(x, (np.ndarray, np.generic)) for x in data) and \
                isinstance(W, (np.ndarray, np.generic))

    def maybeLoadParams(mayeParams):
        if not isinstance(mayeParams, str):
            return mayeParams

        with open(mayeParams, 'r') as fp:
            params = SamplerParams()
            text_format.Parse(fp.read(), params)
            return params

    serializedChains = []
    if checkFromFiles(data, W):
        serializedChains = spmixtures.runSpatialSamplerFromFiles(
            burnin, niter, thin, data, W, params, covariates)

    elif checkFromData(data, W):
        params = maybeLoadParams(params)
        serializedChains = spmixtures.runSpatialSamplerFromData(
            burnin, niter, thin, data, W, params.SerializeToString(),
            covariates)

    else:
        logging.error("Data type not understood")

    return list(map(
        lambda x: getDeserialized(x, UnivariateState), serializedChains))
