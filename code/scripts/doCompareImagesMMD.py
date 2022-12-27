
import sys
import pdb
import numpy as np
import torch
import argparse
import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageset1_filename",
                        help="filename for sampled images", type=str,
                        default="../../data/binarydigits.txt")
    parser.add_argument("--imageset2_filename",
                        help="filename for sampled images", type=str,
                        default="../../results/sampledImages_62722723_n100.csv")
    parser.add_argument("--kernel_type",
                        help="kernel type (multiscale | rbf)", type=str,
                        default="multiscale")
                        # default="rbf")
    args = parser.parse_args()

    imageset1_filename = args.imageset1_filename
    imageset2_filename = args.imageset2_filename
    kernel_type = args.kernel_type

    imageset1 = torch.from_numpy(np.loadtxt(imageset1_filename).T)
    imageset2 = torch.from_numpy(np.loadtxt(imageset2_filename).T)

    mmd = utils.MMD(x=imageset1, y=imageset2, kernel=kernel_type)
    print("MMD: {:.04f}".format(mmd))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
