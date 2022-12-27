
import sys
import pickle
import numpy as np
import configparser
import argparse

sys.path.append("../src/")
import mixtureOfBernoulli

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", type=int,
                        help="estimation result number")
    parser.add_argument('--n_samples', type=int, default=100,
                        help="number of images to sample")
    parser.add_argument(
        "--results_filename_pattern", type=str, help="results filename",
        default="../../results/emMixtureBernoulli_res{:08d}.{:s}")
    parser.add_argument('--sampled_images_filename_pattern',
                        default="../../results/sampledImages_{:d}_n{:d}.csv",
                        help="filename for sampled images pattern")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    N = args.n_samples
    results_filename_pattern = args.results_filename_pattern
    sampled_images_filename_pattern = args.sampled_images_filename_pattern

    estim_res_metadata_filename = \
        results_filename_pattern.format(est_res_number, "metadata")
    estim_res_config = configparser.ConfigParser()
    estim_res_config.read(estim_res_metadata_filename)
    do_not_use_log_prob = \
        estim_res_config["est_params"]["do_not_use_log_prob"] in \
            ["True", "true"]

    model_save_filename = results_filename_pattern.format(est_res_number,
                                                          "pickle")
    with open(model_save_filename, "rb") as f:
        model = pickle.load(f)

    P = model["P"]
    Pi = model["Pi"]
    if not do_not_use_log_prob:
        P = np.exp(P)
        Pi = np.exp(Pi)

    D = P.shape[1]

    mob = mixtureOfBernoulli.MixtureOfBernoulli()
    sampled_images = np.empty((N, D), dtype=np.int32)
    for n in range(N):
        sampled_images[n,:] = mob.sample(Pi=Pi, P=P)

    sampled_images_filename = sampled_images_filename_pattern.format(
        est_res_number, N)
    np.savetxt(sampled_images_filename, sampled_images, fmt="%d")

    breakpoint()

if __name__=="__main__":
    main()
