#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=vaib@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --job-name=base_10_16_1e-5

module purge
module load python3/intel/3.6.3
module load cudnn/9.0v7.3.0.29
source ~/env/bin/activate

python3 roberta_msp.py --roberta_version base --dataset sst2 --in_domain --load_dir /scratch/vag273/nlp/runs/base_10_16_1e-5/out --output_dir /scratch/vag273/nlp/runs/msp_jobs/base_10_16_1e-5
python3 roberta_msp.py --roberta_version base --dataset imdb --load_dir /scratch/vag273/nlp/runs/base_10_16_1e-5/out --output_dir /scratch/vag273/nlp/runs/msp_jobs/base_10_16_1e-5
python3 roberta_msp.py --roberta_version base --dataset counterfactual-imdb --load_dir /scratch/vag273/nlp/runs/base_10_16_1e-5/out --output_dir /scratch/vag273/nlp/runs/msp_jobs/base_10_16_1e-5 --split all
