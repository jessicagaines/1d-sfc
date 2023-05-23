# 1d-sfc

1. Create the virtual environment:

conda env create -f environment.yml

2. Activate the virtual environment:

conda activate pitch_model_env

3. Run a single trial:

python run_pitch_model.py <config file>
python run_pitch_model.py pitch_pert_configs.ini

4. Fit model to data using SBI

jupyter notebook
simulation-based-inference.ipynb