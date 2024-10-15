The repository contains all the implemented codes related to the article "Bayesian Calibration in a multi-output transposition context". 

- investigate_alphamap.py is the implementation of the algorithm to investigate $\boldsymbol{\alpha}_{\text{MAP}}.$
- bayes_lambda.py is dedicated to the MCMC sampling of $\boldsymbol{\lambda}$, for the methods No error, Uniform_error and Hierarchical MAP. 
- bayes_alpha.py is dedicated to the MCMC sample of $\boldsymbol{A}$, for the method Full-bayesian.
- full_bayes.py uses the outputs of bayes_lambda.py and bayes_alpha.py to compute te results of the Full-bayesian approach.
- embedded_discrepancy.py is dedidcated to the method Embedded discrepancy.
- utils_calib.py provides different functions useful for the implementation: Monte Carlo sampling of $\boldsymbol{\lambda}$, computation of likelihoods, normalization of $\boldsymbol{\lambda}$, etc.
- utils_plot_errors.py provides differents functions useful for plotting the results.
- run_all_strategies.ipynb uses the previous .py files to perform the different methods. 
- gp_simus.py is dedicated to the surrogate model.
- plot_summary.ipynb uses the functions of utils_plot_errors.ipynb to plot the different resutls.