import numpy as np

from google.protobuf.internal import encoder
from google.protobuf.internal.decoder import _DecodeVarint32
from scipy.stats import norm

from spatial_mix.protos.py.univariate_mixture_state_pb2 import UnivariateState


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


def _estimateDensity(weights, atoms, xgrid):
    out = np.zeros(len(xgrid))
    for h, atom in enumerate(atoms):
        out += weights[h] * norm.pdf(xgrid, atom.mean, atom.stdev)
    return out


def estimateDensities(chains, xgrid):
    numGroups = len(chains[0].groupParams)
    numIters = len(chains)

    out = []
    for g in range(numGroups):
        curr = np.zeros((numIters, len(xgrid)))
        for i in range(numIters):
            curr[i, :] = _estimateDensity(
                chains[i].groupParams[g].weights, chains[i].atoms, xgrid)

        out.append(curr)
    return out
