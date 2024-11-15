import numpy as np
import pandas as pd
from utils_calib import * 
import sys
sys.modules['scipy.pi'] = np.pi 
sys.modules['scipy.cos'] = np.cos
sys.modules['scipy.sin'] = np.sin 
from pymcmcstat.MCMC import MCMC



# The function MCMC_alpha generates MCMC samples $(\boldsymbol{A})_{i=1}^{N}$, taking as argument 
# - "df_Lambda" a sample $(\boldsymbol{\lambda}_k)_{k=1}^M$ i.i.d. with density $p_{\boldsymbol{\Lambda}}(.\mid \boldsymbol{\alpha}^{\star})$
# - "stored_likelihoods" the associated likelihood
# - "p_lambda_alphastar" the prior densities $p_{\boldsymbol{\Lambda}}(\boldsymbol{\lambda}_k\mid \boldsymbol{\alpha}^\star)_{k=1}^{n}$ computed with $\boldsymbol{\alpha}^\star$
# - "scale" the standard deviation of the truncated normal prior
# - "alpha_star" the estimated maximum a posterior for the hyperparameters $\boldsymbol{\alpha}$, "tune_size" the burnin sample size, "size" the sample size, and "rngseed" the random seed
# - "alpha_min", "alpha_max", "delta_alpha" define the space to explore \mathcal{A}, being on each dimension [(max(alpha_star[ii] - delta_alpha,alpha_min), min(alpha_max, alpha_star[ii]+delta_alpha))]
# - "tune_size" the burnin sample size
# -  "size" the sample size
# - "rngseed" the random seed
# - "index_lambda_p" the index of the parameters lambda that are treated with uniform prior
# - "index_lambda_q" the index of the parameters lambda that are treated with hierarchical description
# - "bMINlambda" the lower bounds of the parameters lambda
# - "bMAXlambda" the upper bounds of the parameters lambda

# For each x_j (defined by idx_loo), the function MCMC_alpha_multichains,:
# – generates "df_Lambda" a sample $(\boldsymbol{\lambda}_k)_{k=1}^M$ i.i.d. with density $p_{\boldsymbol{\Lambda}}(.\mid \boldsymbol{\alpha}^{\star})$
# - Computes "stored_likelihoods" the associated likelihoods 
# - Computes "p_lambda_alphastar" the prior densities $p_{\boldsymbol{\Lambda}}(\boldsymbol{\lambda}_k\mid \boldsymbol{\alpha}^\star)_{k=1}^{n}$ computed with $\boldsymbol{\alpha}^\star$
# - Generates multichains MCMC samples with MCMC_alpha
# - Save the results 

def MCMC_alpha(df_Lambda, stored_likelihoods, p_lambda_alphastar, scale, alpha_star, tune_size, size, alpha_min, alpha_max, delta_alpha, rngseed, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda):

    def ssfun(theta, data): #log likelihood function
        return -2*np.log(max(10**(-30), likelihood_alpha(alpha = theta, likelihoods_alpha_star = stored_likelihoods, denom_is = p_lambda_alphastar, df_Lambda = df_Lambda, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) ))
                         
    bounds = [(max(alpha_star[ii] - delta_alpha,alpha_min), min(alpha_max, alpha_star[ii]+delta_alpha)) for ii in range(len(alpha_star))] #bounds 

    mcstat = MCMC(rngseed=rngseed) 

    x = np.array(range(1))
    y = x.copy()
    mcstat.data.add_data_set(x, y)
    mcstat.simulation_options.define_simulation_options(
        nsimu=int(tune_size+size),
        updatesigma=False, verbosity = 0, waitbar= True)
    mcstat.model_settings.define_model_settings(sos_function=ssfun)
    for ii in range(len(alpha_star)):
        mcstat.parameters.add_model_parameter(
            name=str('$alpha{}$'.format(ii + 1)),
            theta0 = alpha_star[ii],
            minimum = bounds[ii][0],
            maximum = bounds[ii][1]
            )

    mcstat.run_simulation()
    return mcstat.simulation_results.results

def MCMC_alpha_multichains(index_calib, scale, num_chain, tune_size, size, L, alpha_min, alpha_max, delta_alpha, rngseed, results_measures, sigma, myCODE, mm_list, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, pre_path, loo = True, std_code = True):
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(results_measures))
    np.random.seed(rngseed)
    seeds = np.random.randint(1000, size = num_chain) #get seed for each chain
    alpha_df = pd.read_csv(pre_path + f"/calib_{index_calib}/alpha_df.csv", index_col = 0).values
    for idx_loo in list_idx_loo:
        if idx_loo is None: alpha_star = alpha_df[0]
        else: alpha_star = alpha_df[idx_loo]
        np.random.seed(123456)
        df_Lambda = sample_Lambda(alpha = alpha_star, M = L, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q,scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #sample lambda
        Ysimu_list, Ystd_list, stored_likelihoods = get_likelihoods_dflambda(df_Lambda = df_Lambda.values, sigma = sigma, myCODE = myCODE, mm_list = mm_list, results_measures = results_measures, index=[index_calib], std_code = std_code, idx_loo = idx_loo) #get likelihoods
        p_lambda_alphastar = p_lambda_df(df_Lambda, alpha_star, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #get the prior densities with alpha_star
        res = [MCMC_alpha(df_Lambda = df_Lambda, stored_likelihoods = stored_likelihoods, p_lambda_alphastar = p_lambda_alphastar, scale = scale, alpha_star = alpha_star, tune_size = tune_size, size = size,  alpha_min = alpha_min, alpha_max = alpha_max, delta_alpha = delta_alpha, rngseed = ss, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) for ss in seeds] #run every MCMC chain
        samples = np.concatenate([res[i]["chain"][tune_size:,] for i in range(len(res))]) #concatenate the chain without the burnin phase
        if idx_loo is None: save_results(pd.DataFrame(samples), f"samples_alpha_post.csv", pre_path = pre_path, calib = index_calib)
        else: save_results(pd.DataFrame(samples), f"samples_alpha_post_{idx_loo}.csv", pre_path = pre_path, calib = index_calib)
