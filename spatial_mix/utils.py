import logging
import numpy as np
import multiprocessing
import os
import sys

from google.protobuf import text_format
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
from scipy.stats import norm
from scipy.integrate import simps, trapz

from spatial_mix.protos.py.sampler_params_pb2 import SamplerParams
from spatial_mix.protos.py.univariate_mixture_state_pb2 import UnivariateState

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import spmixtures


def loadChains(filename, msgType=UnivariateState):
    """
    Loads an MCMC chain from a .recordio file, returns a list of 'msgType'
    """
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


def writeChains(chains, filename):
    """
    Serializes a list of messages to a file
    """
    with open(filename, "wb") as fp:
        for c in chains:
            try:
                msgStr = c.SerializeToString()
                delimiter = _VarintBytes(len(msgStr))
                fp.write(delimiter + msgStr)
            except Exception as e:
                print(e)
                break


def estimateDensity(weights, means, stdevs, xgrid):
    """
    Estimate the density of a normal mixtures with parameters 
    'weights', 'means' and 'stdevs' over 'xgrid'
    """
    return np.dot(norm.pdf(
        np.hstack([xgrid.reshape(-1, 1)] * len(means)), means, stdevs),
                  weights)


def estimateDensities(chains, xgrids, nproc=-1):
    """
    Given a list of messages representing the MCMC output, evaluates
    the density on 'xgrid' for each step of the chain
    """
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

        eval_normals = norm.pdf(
            np.hstack([xgrids[g].reshape(-1, 1)] * means_chain.shape[0]),
            means_chain,
            stdevs_chain
            ).reshape(len(xgrids[g]), numIters, num_components)
        out.append(np.sum(eval_normals*weights_chain, axis=-1).T)
    return out


def estimateDensitiesRegression(chains, covariates, group):
    """
    Estimates the predictive density for a new sample in areal location 
    'group' with covariate information as 'covariates' given the 
    MCMC output 'chains'
    """
    numIters = len(chains)
    out = []
    num_components = chains[0].num_components
    
    means_chain = np.vstack(
        [list(map(lambda x: x.mean, state.atoms)) for state in chains]
    ).reshape(-1)


    stdevs_chain = np.vstack(
        [list(map(lambda x: x.stdev, state.atoms)) for state in chains]
    ).reshape(-1)

    weights_chain = np.vstack(
        [state.groupParams[group].weights for state in chains])

    regressor_chains = np.vstack(
        [state.regression_coefficients for state in chains])

    reg_means = np.dot(regressor_chains, covs)
    max_err = np.max(np.abs(means_chain) + 5 * np.max(stdevs_chain))
    xgrid = np.linspace(- max_err, max_err, 1000)
    eval_normals = norm.pdf(
        np.hstack([xgrid.reshape(-1, 1)] * means_chain.shape[0]
                ), means_chain, stdevs_chain
    ).reshape(len(xgrid), numIters, num_components)
    eval_dens = np.sum(eval_normals*weights_chain, axis=-1).T + reg_means
    return eval_dens, xgrid


def eval_stan_density(stanfit, xgrid):
    """
    Given a StanFit output, evaluates the density on 'xgrid' for each
    step of the chain
    """
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


def lpml(densities):
    """
    Log Pseudo Marginal Likelihood
    """
    if isinstance(densities, list):
        densities = np.hstack(densities)
    return np.sum(1 / np.mean(1 / densities, axis=0))


def getDeserialized(serialized, objType):
    """
    Utility to de-serialize a message from a string
    """
    out = objType()
    out.ParseFromString(serialized)
    return out


def hellinger_dist(p, q, xgrid):
    """
    Hellinger distance between p and q evaluated on xgrid
    """
    return np.sqrt(0.5 * simps((np.sqrt(p) - np.sqrt(q))** 2, xgrid))


def post_hellinger_dist(estimatedDens, true, xgrid):
    """
    For each step of the chain, computes the Hellinger distance between 
    the density estimate at that step and the true data generating density
    evaluated at 'xgrid' 
    """
    return np.apply_along_axis(
        lambda x: hellinger_dist(x, true, xgrid), 1, estimatedDens)


def kl_div(p, q, xgrid):
    """
    Kullback-Leibler divergence between p and q evaluated on xgrid
    """
    return simps(p * (np.log(p + 1e-5) - np.log(q + 1e-5)), xgrid)


def post_kl_div(estimatedDens, true, xgrid):
    """
    For each step of the chain, computes the KL divergence between 
    the density estimate at that step and the true data generating density
    evaluated at 'xgrid' 
    """
    return np.apply_along_axis(
        lambda x: kl_div(true, x, xgrid), 1, estimatedDens)


def runSpatialMixtureSampler(
        burnin, niter, thin, W, params, data, covariates=[], 
        num_components=None):
    """
    Runs the Gibbs sampler for the SPMIX model for a total of burnin + niter
    iterations, discarding the first 'burnin' ones and keeping in memory
    only one every 'thin' iterations.

    Parameters
    ----------
    burnin: int 
        Number of steps of the burnin
    niter: int
        Number of steps to run *after* the burnin
    thin: int 
        Keep only one every 'thin' iterations
    W: either np.array or string
        If np.array, the proximity matrix
        If string, path to a file (csv) storing the proximity matrix
    params: either SamplerParams or string
        If SamplerParams, the sampler parameters
        If string, path to an 'asciipb' file (human readable) containing 
        the params
    data: either string or list[np.array]
        If string, list to a csv file representing the data
        If list[np.array]: the data. Entry in position 'g' of such list
        contains all the data for group g
    covariates: list[np.array] (optional)
        If not empty, the model will perform a regression on these covariates
    num_components: int (default=None)
        Gives the possibility to override the num_components parameter in
        'params'. Useful b.c. protobuf and pickle (and thus multiprocessing)
        don't mix well.
    """
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
        serializedChains, time = spmixtures.runSpatialSamplerFromFiles(
            burnin, niter, thin, data, W, params, covariates)

    elif checkFromData(data, W):
        params = maybeLoadParams(params)
        if num_components is not None:
            params.num_components = num_components
        serializedChains, time = spmixtures.runSpatialSamplerFromData(
            burnin, niter, thin, data, W, params.SerializeToString(),
            covariates)

    else:
        logging.error("Data type not understood")

    return list(map(
        lambda x: getDeserialized(x, UnivariateState), serializedChains)), time
