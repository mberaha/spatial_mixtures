import numpy as np
import multiprocessing

from scipy.stats import gamma, norm
from spatial_mix.protos.py.univariate_mixture_state_pb2 import HdpState


def estimateDensity(cluster_sizes, betas, atoms, params, xgrid):
    out = np.zeros(len(xgrid))
    weights = np.zeros(len(atoms))
    ntot = sum(cluster_sizes)
    lam = params["priorLambda"]
    mu0 = params["priorMean"]
    a = params["priorA"]
    b = params["priorB"]
    for h in range(len(atoms)):
        weights[h] = cluster_sizes[h] + params["alpha"] * betas[h]

    weights /= (ntot + params["alpha"])

    for h, atom in enumerate(atoms):
        out += weights[h] * norm.pdf(xgrid, atom.mean, atom.stdev)

    bstar = xgrid ** 2 + lam * mu0 + \
        2 * b - (xgrid + (lam * mu0) ** 2 / (1 + lam))

    marg = 1 / (2 * np.pi) * b ** a * a * \
        np.sqrt(lam / (1 + lam)) / (bstar ** (a + 1))

    out += params["alpha"] * betas[-1] / (ntot + params["alpha"]) * marg
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
            params = {
                "alpha": chains[i].alpha,
                "priorMean": chains[i].hyper_params.mu0,
                "priorA": chains[i].hyper_params.a,
                "priorB": chains[i].hyper_params.b,
                "priorLambda": chains[i].hyper_params.lamb}
            curr[i, :] = estimateDensity(
                chains[i].groupParams[g].cluster_size, chains[i].betas,
                chains[i].atoms, params, xgrids[g])

        out.append(curr)
    return out


def getDeserialized(serialized, objType):
    out = objType()
    out.ParseFromString(serialized)
    return out

def runHdpSampler(burnin, niter, thin, data):
    serializedChains = spmixtures.runHdpPythonFromData(
            burnin, niter, thin, data)

    return list(map(
        lambda x: getDeserialized(x, UnivariateState), serializedChains))