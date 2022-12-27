import sys
import argparse
import configparser
import pickle
import numpy as np
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", type=int,
                        help="estimation result number")
    parser.add_argument(
        "--results_filename_pattern", type=str, help="results filename",
        default="../../results/emMixtureBernoulli_res{:08d}.{:s}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    results_filename_pattern = args.results_filename_pattern

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
    K = P.shape[0]
    D = P.shape[1]

    P_image_filename_pattern = "../../figures/{:08d}_P_K{:d}.{:s}"
    for k in range(K):
        fig = go.Figure()
        img = P[k, :].reshape(int(np.sqrt(D)), int(np.sqrt(D)))
        trace = go.Heatmap(z=img, colorscale="gray")
        fig.add_trace(trace)
        fig.update_yaxes(autorange="reversed")
        fig.write_image(P_image_filename_pattern.format(est_res_number, k,
                                                        "png"))
        fig.write_html(P_image_filename_pattern.format(est_res_number, k,
                                                       "html"))

    R_image_filename_pattern = "../../figures/{:08d}_R.{:s}"
    R = model["R"]
    fig = go.Figure()
    trace = go.Heatmap(z=R, colorscale="gray")
    fig.add_trace(trace)
    fig.write_image(R_image_filename_pattern.format(est_res_number, "png"))
    fig.write_html(R_image_filename_pattern.format(est_res_number, "html"))

    Pi_image_filename_pattern = "../../figures/{:08d}_Pi.{:s}"
    fig = go.Figure()
    trace = go.Bar(x=np.arange(len(Pi)), y=Pi)
    fig.add_trace(trace)
    fig.update_xaxes(title_text="Component index")
    fig.update_yaxes(title_text="Coefficient value")
    fig.write_image(Pi_image_filename_pattern.format(est_res_number, "png"))
    fig.write_html(Pi_image_filename_pattern.format(est_res_number, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
