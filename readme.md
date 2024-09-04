The repository contains all the implemented codes. 

- investigate_alphamap.ipynb is the implementation of the algorithm to investigate $\boldsymbol{\alpha}_{\text{MAP}}.$
- bayes_lambda.ipynb is dedicated to the MCMC sampling of $\boldsymbol{\lambda}$, for the methods No error, Uniform_error and Hierarchical MAP.
- bayes_alpha.ipynb is dedicated to the MCMC sample of $\boldsymbol{A}$, for the method Full-bayesian.
- bayes_alpha.ipynb uses the outputs of bayes_lambda.ipynb and bayes_alpha.ipynb to compute te results of the Full-bayesian approach.
- embedded_discrepancy.ipynb is dedidcated to the method Embedded discrepancy.
- utils_calib.ipynb provides different functions useful for the implementation: Monte Carlo sampling of $\boldsymbol{\lambda}$, computation of likelihoods, normalization of $\boldsymbol{\lambda}$, etc.
- utils_plot_errors.ipynb provides differents functions useful for plotting the results.
- run_all_strategies.ipynb runs the notebooks for each strategy for a given set of parameters.
- gp_simus.ipynb is dedicated to the surrogate model 
- plot_summary.ipynb uses the functions of utils_plot_errors.ipynb to plot the different resutls.

The repositories starting with "seedx_" are the results of the different methods, each one is associated with a different design $\mathbb{X}$, and contains different folders that correspond to the investigated approaches. 
