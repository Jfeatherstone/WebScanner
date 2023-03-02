#!/bin/bash

#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --mem=384G
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --job-name=WebScan
#SBATCH --output=Name_%J.out
#SBATCH --mail-type ALL
#SBATCH --mail-user jack.featherstone@oist.jp

source "/home/j/john-featherstone/.bashrc"
export PYTHONPATH="/home/j/john-featherstone/apps/lib/python3.9/site-packages/"
conda activate ml

#python3 scatter_threads.py "/bucket/BandiU/Jack/DATA/WebScanning/2023-02-28_A" -o "images/2023-02-28_A_neighbors.gif" --ds 100 --neighbors
#python3 scatter_threads.py "/bucket/BandiU/Jack/DATA/WebScanning/2023-02-28_B" -o "images/2023-02-28_B_neighbors.gif" --ds 100 --neighbors
python3 scatter_threads.py "/bucket/BandiU/Jack/DATA/WebScanning/2023-02-28_C" -o "images/2023-02-28_C_neighbors.gif" --ds 300 --neighbors
python3 scatter_threads.py "/bucket/BandiU/Jack/DATA/WebScanning/2023-02-28_D" -o "images/2023-02-28_D_neighbors.gif" --ds 300 --neighbors
