import sys
import argparse
import configparser
import pickle
import numpy as np

sys.path.append("../src/")
import mixtureOfBernoulli

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("error")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", type=int,
                        help="estimation result number")
    parser.add_argument(
        "--images_filename", type=str, help="images filename",
        default="../../data/test_mnist_images_digits2_3_4_numPerDigit100.csv")
    parser.add_argument(
        "--estimated_model_filename_pattern", type=str,
        help="estimated model filename pattern",
        default="../../results/emMixtureBernoulli_res{:08d}.{:s}")
    parser.add_argument(
        "--cv_ll_eval_filename_pattern", type=str,
        help="cross validated log-likelihood evalution filename pattern",
        default="../../results/cv_ll_eval_{:08d}.txt")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    images_filename = args.images_filename
    estimated_model_filename_pattern = args.estimated_model_filename_pattern
    cv_ll_eval_filename_pattern = args.cv_ll_eval_filename_pattern

    images = np.loadtxt(images_filename)

    estim_res_metadata_filename = \
        estimated_model_filename_pattern.format(est_res_number, "metadata")
    estim_res_config = configparser.ConfigParser()
    estim_res_config.read(estim_res_metadata_filename)
    do_not_use_log_prob = \
        estim_res_config["est_params"]["do_not_use_log_prob"] in \
            ["True", "true"]

    model_save_filename = estimated_model_filename_pattern.format(
        est_res_number, "pickle")
    with open(model_save_filename, "rb") as f:
        model = pickle.load(f)

    mob = mixtureOfBernoulli.MixtureOfBernoulli()
    if do_not_use_log_prob:
        Pi = model["Pi"]
        P = model["P"]
        ll = mob.computeLogLikelihoodWithNonLogProbs(x=images, Pi=Pi, P=P)
    else:
        logPi = model["Pi"]
        logP = model["P"]
        ll = mob.computeLogLikelihoodWithLogProbs(x=images, logPi=logPi, logP=logP)

    cv_ll_eval_filename = cv_ll_eval_filename_pattern.format(est_res_number)
    with open(cv_ll_eval_filename, "w") as f:
        f.writelines(f"{ll}")

    print(ll)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
