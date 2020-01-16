import numpy as np
import multiprocessing
import os
import sys

from google.protobuf import text_format
from google.protobuf.internal.decoder import _DecodeVarint32
from scipy.stats import norm

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


def estimateDensity(weights, atoms, xgrid):
    out = np.zeros(len(xgrid))
    for h, atom in enumerate(atoms):
        out += weights[h] * norm.pdf(xgrid, atom.mean, atom.stdev)
    return out


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
    for g in range(numGroups):
        curr = np.zeros((numIters, len(xgrids[g])))
        for i in range(numIters):
            curr[i, :] = estimateDensity(
                chains[i].groupParams[g].weights, chains[i].atoms, xgrids[g])

        out.append(curr)
    return out


def lpml(densities):
    if isinstance(densities, list):
        densities = np.hstack(densities)
    return np.sum(1 / np.mean(1 / densities, axis=0))


def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out


def runSpatialMixtureSamplerFromFiles(
        burnin, niter, thin, data_file, W_file, params_file):

    serializedChains = spmixtures._runSpatialSamplerFromFiles(
        burnin, niter, thin, data_file, W_file, params_file)

    return list(map(
        lambda x: getDeserialized(x, UnivariateState), serializedChains))


def runSpatialMixtureSamplerFromData(
        burnin, niter, thin, data, W, params):

    if isinstance(params, str):
        with open(params, 'r') as fp:
            params = SamplerParams()
            text_format.Parse(fp.read(), params)

    serializedChains = spmixtures._runSpatialSamplerFromData(
        burnin, niter, thin, data, W, params.SerializeToString())

    out = list(map(
        lambda x: getDeserialized(x, UnivariateState), serializedChains))
    return out
