import sys
import argparse
import pandas as pd
import plotly.graph_objs as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv_ll_eval_filename_pattern", type=str,
        help="cross validation log-likelihood evaluation filename pattern",
        default="../../results/cv_ll_eval_{:08d}.txt")
    parser.add_argument(
        "--relevant_models_filename", type=str,
        help="relevant models filename",
        default="../../notes/relevantModelsCrossValidated.txt")
    parser.add_argument(
        "--figure_filename_pattern", type=str,
        help="figure filename pattern",
        default="../../figures/llsVsKCrossValidated.{:s}")
    args = parser.parse_args()

    relevant_models_filename = args.relevant_models_filename
    cv_ll_eval_filename_pattern = args.cv_ll_eval_filename_pattern
    figure_filename_pattern = args.figure_filename_pattern

    relevant_models = pd.read_csv(relevant_models_filename, sep=",")
    n_models = relevant_models.shape[0]
    ks = [None for i in range(n_models)]
    lls = [None for i in range(n_models)]
    for i in range(n_models):
        est_res_number = relevant_models.loc[i, "est_res_number"]
        ks[i] = relevant_models.loc[i, "k"]
        cv_ll_eval_filename = cv_ll_eval_filename_pattern.format(
            est_res_number)
        with open(cv_ll_eval_filename, "r") as f:
            lls[i] = float(f.readline())
    fig = go.Figure()
    trace = go.Scatter(x=ks, y=lls)
    fig.add_trace(trace)
    fig.update_xaxes(title_text="K")
    fig.update_yaxes(title_text="Log-Likelihood")
    fig.write_image(figure_filename_pattern.format("png"))
    fig.write_html(figure_filename_pattern.format("html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
