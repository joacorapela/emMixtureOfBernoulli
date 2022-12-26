import sys
import os
import pickle
import argparse
import random
import configparser
import numpy as np

sys.path.append("../src/")
import emMixtureOfBernoulli


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("n_components", type=int,
                        help="number of EM components")
    parser.add_argument("--do_not_use_log_prob", action="store_true",
                        help="do not use log probabilities")
    parser.add_argument("--tol", type=float,
                        help="EM tolerance", default=1e-5)
    parser.add_argument("--max_iter", type=int,
                        help="EM maximum number of iterations", default=100)
    parser.add_argument("--random_seed", type=int,
                        help="random seed", default=-1)
    parser.add_argument("--images_filename", type=str, help="images filename",
                        default="../../data/binarydigits.txt")
    parser.add_argument(
        "--results_filename_pattern", type=str, help="results filename",
        default="../../results/emMixtureBernoulli_res{:08d}.{:s}")
    args = parser.parse_args()

    K = args.n_components
    do_not_use_log_prob = args.do_not_use_log_prob
    tol = args.tol
    max_iter = args.max_iter
    random_seed = args.random_seed
    images_filename = args.images_filename
    results_filename_pattern = args.results_filename_pattern

    if random_seed > 0:
        np.random.seed(seed=random_seed)
    images = np.loadtxt(images_filename)
    D = images.shape[1]

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estim_res_metadata_filename = \
            results_filename_pattern.format(estResNumber, "metadata")
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    modelSaveFilename = results_filename_pattern.format(estResNumber, "pickle")

    estim_res_config = configparser.ConfigParser()
    estim_res_config["est_params"] = {
        "n_components ": K,
        "do_not_use_log_prob": do_not_use_log_prob,
        "tol": tol,
        "max_iter": max_iter,
        "random_seed": random_seed,
        "images_filename": images_filename,
    }
    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)

    # P0.dtype == float64
    P0 = np.random.uniform(low=0.25, high=0.75, size=(K, D))
    normalizer = np.sum(P0, 0)  # \in D
    P0 = P0 / np.expand_dims(normalizer, 0)

    Pi0 = np.ones(K, dtype=np.float64)/K

    if do_not_use_log_prob:
        em = emMixtureOfBernoulli.EMmixtureOfBernoulliNonLogProb(x=images)
    else:
        em = emMixtureOfBernoulli.EMmixtureOfBernoulliLogProb(x=images)
    Pi, P, R, lls = em.optimize(Pi0=Pi0, P0=P0, tol=tol, max_iter=max_iter)

    resultsToSave = {"Pi": Pi,
                     "P": P,
                     "R": R,
                     "lls": lls}
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
