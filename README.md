# ood-detection
Analyzing OOD detection in text.


## Usage

### create_sub_scripts.py
Creates all the sbatch submission jobs in the submissions_scripts directory. Use sbatch to submit those to the cluster. 


## Dependencies for roberta_fine_tune.py
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
pip install transformers
pip install datasets
pip install -Iv 'dill==0.3.3' --force-reinstall
pip install -Iv 'pandas==1.1.4' --force-reinstall
