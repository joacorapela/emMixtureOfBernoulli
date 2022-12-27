#!/bin/csh

if ($#argv <> 2) then
    echo "you must give exactly two parameters"
else
    set est_res_number = $argv[1]
    set N = $argv[2]
endif

ipython --pdb doPlotBinaryImages.py -- \
    --binaryImagesFilename ../../results/sampledImages_{$est_res_number}_n{$N}.csv \
    --plotFilename ../../figures/sampledImages_{$est_res_number}_n{$N}.png
