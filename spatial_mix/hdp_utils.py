import numpy as np
import multiprocessing

from scipy.stats import gamma, norm


def estimateDensity(cluster_sizes, betas, atoms, params, xgrid):
    out = np.zeros(len(xgrid))
    weights = np.zeros(len(atoms))
    ntot = sum(cluster_sizes)
    for h in range(len(atoms)):
        weights[h] = cluster_sizes[h] + params["alpha"] * betas[h]

    weights /= (ntot + params["alpha"])

    for h, atom in enumerate(atoms):
        out += weights[h] * norm.pdf(xgrid, atom.mean, atom.stdev)

    # TODO: rough approximation, still better than nothing
    tau = gamma.rvs(params["priorA"], 1 / params["priorB"])
    mean = norm.rvs(
        params["priorMean"], 1.0 / np.sqrt(params["priorLambda"] * tau))
    out += params["alpha"] * betas[-1] / (ntot + params["alpha"]) * norm.pdf(
        xgrid, mean, 1.0 / np.sqrt(tau))
    # weights[len(atoms)] = params["alpha"] * betas[len(atoms)]
    # out += weights[len(atoms)] * norm.pdf(xgrid, params)
    return out


def _aux(state, g, xgrid):
    return estimateDensity(
        state.groupParams[g].weights, state.atoms, xgrid)


def estimateDensities(chains, xgrid, nproc=-1):

    numGroups = len(chains[0].groupParams)
    numIters = len(chains)
    if nproc == -1:
        nproc = multiprocessing.cpu_count() - 1

    out = []
    for g in range(numGroups):
        curr = np.zeros((numIters, len(xgrid)))
        for i in range(numIters):
            params = {
                "alpha": chains[i].alpha,
                "priorMean": chains[i].hyper_params.mu0,
                "priorA": chains[i].hyper_params.a,
                "priorB": chains[i].hyper_params.b,
                "priorLambda": chains[i].hyper_params.lamb}
            curr[i, :] = estimateDensity(
                chains[i].groupParams[g].cluster_size, chains[i].betas,
                chains[i].atoms, params, xgrid)

        out.append(curr)
    return out
