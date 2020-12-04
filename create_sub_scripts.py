#!/usr/bin/env python3

command_list = ["#!/bin/bash",
"#",
"#SBATCH --nodes=1",
"#SBATCH --ntasks-per-node=1",
"#SBATCH --cpus-per-task=2",
"#SBATCH --time=24:00:00",
"#SBATCH --mem=32GB",
"#SBATCH --gres=gpu:1",
"#SBATCH --job-name=nlp_project",
"#SBATCH --mail-type=END",
"##SBATCH --mail-user=apo249@nyu.edu",
"#SBATCH --output=slurm_%j.out",
"",
"",
"module purge",
"module load python3/intel/3.6.3 cudnn/9.0v7.3.0.29",
"source ~/pyenv/py3.6.3/bin/activate"]

task_names = ["sst2", "imdb"]
roberta_versions = ["roberta-base", "roberta-large"]
num_epochs = [3, 10]
max_seq_lengths = [128]
batch_sizes = [16, 32]
learning_rates = ["3e-6", "1e-5", "3e-5", "5e-5"]

for task in task_names:
    for epoch in num_epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for roberta_version in roberta_versions:
                    prefix = roberta_version + "_" + str(task) + "_" + str(epoch) + "_" + str(batch_size) + "_" + learning_rate 
                    job_file = "submission_scripts/" + prefix + ".sh"
                    cache_dir = "caches/" + prefix + "_cache"
                    output_dir = "outputs/" + prefix + "_output"
              
                    command = "python roberta_fine_tune.py --task_name " +  task + " --roberta_version " + roberta_version + " --cache_dir " + cache_dir + " --num_epochs " + str(epoch) + " --batch_size " + str(batch_size) + " --learning_rate " + learning_rate + " --output_dir " + output_dir

                    with open(job_file, "w") as f:
                        for cmd in command_list:
                            f.write(cmd + "\n")
                        f.write(command + "\n")


