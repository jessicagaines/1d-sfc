# 1d-sfc

This code implements in Python a one-dimensional state feedback control model as 
described in Houde & Nagarajan (2011) and implemented in Matlab in
Houde et al. (2014). Also included is a script to fit the 
SFC model to empirical pitch perturbation data using the SBI package 
(Cranmer et al., 2020; Tejero-Cantero et al., 2020; https://sbi-dev.github.io/sbi/)

1. Create the virtual environment:

```python
conda env create -f environment.yml
```

2. Activate the virtual environment:

```
conda activate 1d_sfc_env
```

3. Run a single trial:

```
python run_pitch_model.py [config file]
```
Automatically set the observer's noise estimate to the value of feedback noise in the plant
```
python run_pitch_model.py pitch_pert_configs.ini 
```
Independently vary the observer's noise estimate and the value of feedback noise in the plant
```
python run_pitch_model.py pitch_pert_configs_w_est.ini 
```

4. Fit model to data using SBI
```
python simulation-based-inference.py [int # training simulations] [int # repetitions of training] [boolean generate new training data?] [int index of parameter to ablate if any]
```
Run 10 repetitions of the inference procedure and store the posterior functions as .pkl files
```
python simulation-based-inference.py 100000 10 True
```
Use previously stored posterior functions from 10 repetitions -- rerun sampling only
```
python simulation-based-inference.py 100000 10 False
```
Ablate parameter 3 (indexing starts at 0) and run inference procedure. Fix parameter 3 at the inferred value for the first data set from the full model (therefore the inference procedure must be run for the full model first before running ablations)
```
python simulation-based-inference.py 100000 10 True 3
```

References


Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. Proceedings of the National Academy of Sciences 117(48), 30055-30062. https://doi.org/10.1073/pnas.1912789117.


Tejero-Cantero, A., et al. (2020). Sbi: A toolkit for simulation-based inference. Journal of Open Source Software 5(52), 2505. https://doi.org/10.21105/joss.02505


Houde, J.F. & Nagarajan, S.S. (2011). Speech production as state feedback control. Frontiers in Human Neuroscience 5(82), 1-14. https://doi.org/10.3389/fnhum.2011.00082.


Houde, J.F., Niziolek, C.A., Kort, N., Agnew, Z., & Nagarajan, S.S. (2014, May 5-8). Simulating a state feedback model of speaking. 10th International Seminar on Speech Production, Cologne, Germany.https://www.researchgate.net/publication/284486657_Simulating_a_state_feedback_model_of_speakisp
