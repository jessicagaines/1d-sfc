# 1d-sfc

This code implements a one-dimensional state feedback control model as 
described in Houde & Nagarajan (2011) and Houde et al. (2014). Also 
included is a Jupyter notebook to fit the SFC model to empirical 
pitch perturbation data using the SBI package (Cranmer et al., 2020; 
Tejero-Cantero et al., 2020; https://www.mackelab.org/sbi/)

1. Create the virtual environment:

```python
conda env create -f environment.yml
```

2. Activate the virtual environment:

```
conda activate pitch_model_env
```

3. Run a single trial:

```
python run_pitch_model.py [config file]
```
```
python run_pitch_model.py pitch_pert_configs.ini
```

4. Fit model to data using SBI
```
jupyter notebook
simulation-based-inference.ipynb
```

References
Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. Proceedings of the National Academy of Sciences 117(48), 30055-30062. https://doi.org/10.1073/pnas.1912789117.
Tejero-Cantero, A., et al. (2020). Sbi: A toolkit for simulation-based inference. Journal of Open Source Software 5(52), 2505. https://doi.org/10.21105/joss.02505
Houde, J.F. & Nagarajan, S.S. (2011). Speech production as state feedback control. Frontiers in Human Neuroscience 5(82), 1-14. https://doi.org/10.3389/fnhum.2011.00082.
Houde, J.F., Niziolek, C.A., Kort, N., Agnew, Z., & Nagarajan, S.S. (2014, May 5-8). Simulating a state feedback model of speaking. 10th International Seminar on Speech Production, Cologne, Germany.https://www.researchgate.net/publication/284486657_Simulating_a_state_feedback_model_of_speakisp
