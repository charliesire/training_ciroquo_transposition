import numpy as np
import pandas as pd
from utils_calib import * 
import sys
sys.modules['scipy.pi'] = np.pi 
sys.modules['scipy.cos'] = np.cos
sys.modules['scipy.sin'] = np.sin 
from pymcmcstat.MCMC import MCMC


# The function plot_transpo computes the prediction mean and standard deviation, from the list of GP means and standard deviations obtained for each posterior sample lambda_k

# The function compute_error computes the RMSRE and the levels associated with the prediction intervals

# The function MCMC_lambda generates MCMC samples $(\boldsymbol{\lambda})_{k=1}^{M}$, taking as argument 
# - "index_calib" the index $t$ of the observed variable $y_t$, 
# - "model_error" a boolean indicating if the parameters representing the model error must be included or not
# - "scale" the standard deviation of the truncated normal prior, 
# - alpha_map the estimated maximum a posterior for the hyperparameters $\boldsymbol{\alpha}$, 
# - "idx_loo" the index of the observation $x_j$ that must be removed in the LOO scheme, 
# - "tune_size" the burnin sample size, 
# - "size" the sample size
# - "mm_list" the parameters of the GP metamodel
# - "results_measures" the dataframe of the observations
# - "sigma" the std deviation of the observation noise
# - "rngseed" the random seed
# - "index_lambda_p" the index of the parameters lambda that are treated with uniform prior
# - "index_lambda_q" the index of the parameters lambda that are treated with hierarchical description
# - "bMINlambda" the lower bounds of the parameters lambda
# - "bMAXlambda" the upper bounds of the parameters lambda

# The function MCMC_lambda_multichains generate multichains MCMC samples for each x_j, and then save the results: the posterior samples and the associated outputs

# The function bayes_lambda_results compute the performance metrics from the obtained posterior outputs

def plot_transpo(Ysimu_list, Ystd_list = None):
    nb_outputs = Ysimu_list[0].shape[1]
    nb_exp = Ysimu_list[0].shape[0]
    Y_mean = pd.DataFrame(np.zeros((nb_exp, nb_outputs)))
    Y_std = pd.DataFrame(np.zeros((nb_exp,nb_outputs)))
    for simu in range(1,nb_exp+1): #for each observation point x_j
        for index_predict in range(1,nb_outputs+1): #for each output variable y_t
            list_simus = [Ysimu_list[k].iloc[simu-1, index_predict-1] for k in range(len(Ysimu_list))] #get all the output values (list of length M)
            Y_mean.iloc[simu-1, index_predict-1] = np.mean(list_simus) #get prediction mean
            Y_std.iloc[simu-1, index_predict-1] = np.sqrt(np.var(list_simus)) #get prediction std (without considering std deviation of GP)
            if not (Ystd_list is None):  
                list_var = [Ystd_list[k].iloc[simu-1, index_predict-1]**2 for k in range(len(Ystd_list))] 
                Y_std.iloc[simu-1, index_predict-1] = np.sqrt(Y_std.iloc[simu-1, index_predict-1]**2 + np.mean(list_var))  #Add std deviation of GP                                              
    return pd.concat([Y_mean, Y_std])


def compute_error(Ysimu_list, true_values, Ystd_list = None):
    nb_outputs = Ysimu_list[0].shape[1] 
    nb_exp = Ysimu_list[0].shape[0]
    res = []
    res_intervals = np.empty((nb_exp,0))
    for idx in np.array(range(1,nb_outputs+1)): #for each output variable y_t
        simus = np.concatenate([[Ysimu_list[k].iloc[:,idx-1].values] for k in range(len(Ysimu_list))]) #get all the output values (dim M x n)
        if Ystd_list is not None:
            stds = np.concatenate([[Ystd_list[k].iloc[:,idx-1].values] for k in range(len(Ystd_list))]) # get all the stds values (dim M x n)
        pred = np.apply_along_axis(np.mean, 0,simus) #compute the prediction means
        error = dist2(pred,true_values[f"Y{idx}"]) #compute the rmsre
        res.append(error) 
        eta = abs(pred-true_values[f"Y{idx}"]).values
        if Ystd_list is not None:
            intervals = np.apply_along_axis(np.mean, 0, norm.cdf((pred+eta-simus)/stds) - norm.cdf((pred-eta-simus)/stds)).reshape(-1,1) #compute the levels of the predictions intervals
        else: intervals = np.apply_along_axis(np.mean, 0, abs(pred-simus)<eta).reshape(-1,1) 
        res_intervals = np.concatenate([res_intervals, intervals], axis=1)
    return res, res_intervals

def MCMC_lambda(index_calib, model_error, scale, alpha_map, idx_loo, tune_size, size, myCODE, mm_list, results_measures, sigma, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, rngseed = None, std_code = True): 
    
    def ssfun(theta, data): #log likelihood function
        xdata = data.xdata[0]
        ydata = data.ydata[0]
        lambd = transform_Lambda(Lambda = theta, index_lambda_p= index_lambda_p, index_lambda_q =index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) 
        if std_code:
            Ysimu, Ystd = myCODE(lambd, index = [index_calib],  std_bool = std_code, vectorize = False, idx_loo = idx_loo, mm_list = mm_list) #get the output for all observations points x_j except at idx_loo
            ss = np.prod(norm.pdf(ydata[:,0]-Ysimu.values.flatten(), loc=0, scale=np.sqrt(sigma**2 + Ystd.values.flatten()**2))) #compute the likelihood
        else: 
            Ysimu = myCODE(lambd, index = [index_calib],  std_bool = std_code, vectorize = False, idx_loo = idx_loo, mm_list = mm_list) #get the output for all observations points x_j except at idx_loo
            ss = np.prod(norm.pdf(ydata[:,0]-Ysimu.values.flatten(), loc=0, scale=np.sqrt(sigma**2))) #compute the likelihood
        return -2*np.log(ss)

    mcstat = MCMC(rngseed=rngseed)

    x = np.array(list(set(range(len(results_measures))) - set([idx_loo]))) 
    y = results_measures[list(set(range(len(results_measures))) - set([idx_loo]))]#all measures points except loo
    mcstat.data.add_data_set(x, y)
    mcstat.simulation_options.define_simulation_options(
        nsimu=int(tune_size+size),
        updatesigma=False, verbosity = 0, waitbar= True)
    mcstat.model_settings.define_model_settings(sos_function=ssfun)
    
    
    for ii in range(len(index_lambda_p)):
        
        mcstat.parameters.add_model_parameter(
            name=str('$lambd_p_{}$'.format(ii + 1)),
            theta0=0.5,
            minimum=0,
            maximum=1
            ) #uniform prior
    
    if(len(index_lambda_q)) > 0:
        if model_error:
            for ii in range(len(index_lambda_q)):
                mcstat.parameters.add_model_parameter(
                    name=str('$lambd_q_{}$'.format(ii + 1)),
                    theta0=0.5,
                    sample = True, 
                    minimum = 0,
                    maximum = 1, 
                    prior_mu=alpha_map[ii],
                    prior_sigma = scale
                    ) #truncated gaussian prior
        else:
            for ii in range(len(index_lambda_q)):
                mcstat.parameters.add_model_parameter(
                    name=str('$lambd_q_{}$'.format(ii + 1)),
                    theta0=0.5,
                    sample = False
                    ) # if no model error, value of numerical parameters fixed at 0.5
    mcstat.run_simulation()
    return mcstat.simulation_results.results


def MCMC_lambda_multichains(index_calib, model_error, scale, num_chain, tune_size, size, myCODE, mm_list, results_measures, sigma, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, rngseed, pre_path, nb_outputs, loo = True, std_code = True):
    np.random.seed(rngseed)
    seeds = np.random.randint(1000, size = num_chain) #get a random seed for each chain
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(results_measures))
    alpha_map = None
    list_res = []
    list_res_std = []
    for idx_loo in list_idx_loo: #for each observation point x_j
        if (len(index_lambda_q) > 0) & model_error: 
            alpha_df = pd.read_csv(pre_path + f"/calib_{index_calib}/alpha_df.csv", index_col = 0).values
            if idx_loo is None: alpha_map = alpha_df[0]
            else: alpha_map = alpha_df[idx_loo]
        res = [MCMC_lambda(index_calib = index_calib, model_error = model_error, scale = scale, alpha_map = alpha_map, idx_loo = idx_loo, tune_size = tune_size, size = size, myCODE = myCODE, mm_list = mm_list, results_measures = results_measures, sigma = sigma, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, rngseed = ss, std_code = std_code) for ss in seeds] #run every MCMC chain

        samples = np.concatenate([res[i]["chain"][tune_size:,] for i in range(len(res))]) #concatenate the MCMC samples without the burnin phase
        if not model_error: samples = np.concatenate([samples, np.array([res[0]['theta'][len(index_lambda_p):]]*len(samples))], axis = 1)
        
        lambd_post = np.apply_along_axis(lambda x:transform_Lambda(Lambda = x, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda), 1, samples) #back to physical values 
        
        if idx_loo is None: save_results(pd.DataFrame(lambd_post), f"lambd_post.csv", pre_path = pre_path, calib = index_calib) #save MCMC samples
        else: save_results(pd.DataFrame(lambd_post), f"lambd_post_{idx_loo}.csv", pre_path = pre_path, calib = index_calib) #save MCMC samples
        
        if not idx_loo is None: 
            if std_code:
                res, res_std = myCODE(lambd = lambd_post, index = np.array(range(1,nb_outputs+1)), std_bool = std_code, vectorize = True, idx_loo = idx_loo, new_x = True, mm_list = mm_list) #compute the 3 outputs at idx_loo
                list_res.append(res)
                list_res_std.append(res_std) 
            else: 
                res = myCODE(lambd = lambd_post, index = np.array(range(1,nb_outputs+1)),  std_bool = std_code, vectorize = True, idx_loo = idx_loo, new_x = True, mm_list = mm_list) #compute the 3 outputs at idx_loo
                list_res.append(res)
    
    if loo: 
        Ysimu_list = [pd.DataFrame(np.concatenate([list_res[ii].iloc[k,:] for ii in range(len(list_res))]).reshape(len(list_res),nb_outputs)) for k in range(len(list_res[0]))] #concatenate all outputs
        if std_code: Ystd_list = [pd.DataFrame(np.concatenate([list_res_std[ii].iloc[k,:] for ii in range(len(list_res_std))]).reshape(len(list_res_std),nb_outputs)) for k in range(len(list_res_std[0]))] #concatenate all standard deviation 
    else:
        if std_code: Ysimu_list,Ystd_list = myCODE(lambd_post, index = [kk for kk in range(1, nb_outputs+1)],  std_bool = True, vectorize = True, idx_loo = None,  mm_list = mm_list) #compute all outputs
        else: Ysimu_list = myCODE(lambd_post, index = [kk for kk in range(1, nb_outputs+1)],  std_bool = False, vectorize = True, idx_loo = None,  mm_list = mm_list) #compute all outputs
    save_results(pd.concat(Ysimu_list), "Ysimu_list.csv", pre_path = pre_path, calib = index_calib) #save outputs
    if std_code: save_results(pd.concat(Ystd_list), "Ystd_list.csv", pre_path = pre_path, calib = index_calib) #save std deviations 
        
def bayes_lambda_results(index_calib, pre_path, true_values, std_code):
    YY = pd.read_csv(pre_path + f"/calib_{index_calib}/Ysimu_list.csv", index_col = 0) #get the output values
    Ysimu_list = np.array_split(YY, len(YY) // len(true_values))# change the format to a list of dataframes: one df for each lambda
    if std_code: 
        Ystd = pd.read_csv(pre_path + f"/calib_{index_calib}/Ystd_list.csv", index_col = 0) #get the stds values
        Ystd_list = np.array_split(Ystd, len(Ystd) // len(true_values))# change the format to a list of dataframes: one df for each lambda
    else: Ystd_list = None
    plot1 = plot_transpo(Ysimu_list = Ysimu_list, Ystd_list = Ystd_list) #compute prediction mean and standard deviation
    save_results(plot1, "plot_alpha_map.csv",pre_path = pre_path, calib = index_calib) #save results
    errors, intervals = compute_error(Ysimu_list = Ysimu_list, true_values = true_values, Ystd_list = Ystd_list) #compute RMSRE and p^0.9
    save_results(pd.DataFrame(errors), "errors_map.csv", pre_path = pre_path, calib = index_calib) #save results
    save_results(pd.DataFrame(intervals), "conf_level_map.csv", pre_path = pre_path, calib = index_calib) #save results
