"""
This scripts generates the data for the somulated example #1

Usage: from the root folder (../), run
python3 -m scripts.generate_simulation1_data \
    --output_path=<path_to_output_folder>
"""

import argparse
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd

from scipy.stats import t, skewnorm, cauchy, chi2

np.random.seed(124151)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rep", type=int, default=100)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()


    outdir = os.path.join(args.output_path, "datasets")
    os.makedirs(outdir, exist_ok=True)

    for j in range(3):
        out = os.path.join(outdir, "scenario{0}".format(j))
        os.makedirs(out, exist_ok = True)

    for i in range(args.num_rep):
        datas = [
            simulate_data_scenario12(1000, 1000),
            simulate_data_scenario12(1000, 10),
            simulate_data_scenario3(100, 100)
        ]

        for j in range(3):
            datas[j].to_csv(
                os.path.join(outdir, "scenario{0}/rep{1}.csv".format(j, i)))
