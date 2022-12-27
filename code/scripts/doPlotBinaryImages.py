"""Assignment 1, Unsupervised Learning, UCL 2003
Author: Zoubin Gahramani
Ported to Python by Raza Habib and Jamie Townsend 2017"""
import pdb
import argparse
import numpy as np
from matplotlib import pyplot as plt
import utils

# Python comments use a hash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binaryImagesFilename', default="../results/mapSampledImages.csv", help="filename containing the binary images")
    parser.add_argument('--plotFilename', default="../figures/mapSampledImages.png", help="plot filename")
    parser.add_argument('--figWidth', type=int, default=5, help="figure width (inches)")
    parser.add_argument('--figHeight', type=int, default=5, help="figure height (inches)")
    args = parser.parse_args()

    binary_images_filename = args.binaryImagesFilename
    plot_filename = args.plotFilename
    fig_width = args.figWidth
    fig_height = args.figHeight

    binary_images = np.loadtxt(binary_images_filename)
    N, D = binary_images.shape

    plt.figure(figsize=(fig_width, fig_height))
    for n in range(N):
        plt.subplot(np.sqrt(N), np.sqrt(N), n+1)
        utils.display_one_digit_fig(digit_vector=binary_images[n, :], show_colorbar=False)
    plt.savefig(plot_filename)
    plt.show()
    pdb.set_trace()

if __name__ == "__main__":
    main()
