import os, sys
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#The function plot_mean_std plots the mean and standard deviations for each method, and compares it to the true values and the calibration measures. 

#The function compare_errors gathers the RMSRE and levels of prediction intervals of all method, and compute $p^{0.9}_{N,M}$.

#The function plot_errors plots the RMSRE and $p^{0.9}_{N,M}$ for each method.

variable_names = [r"$\ell_{F} - \ell_{0}$", r"$r_{F}$", r"$\epsilon_{max}$"] #names of the three outputs

def plot_mean_std(index_calib, results_measures, true_values, sigma, pre_path, no_error = False, unif_error = False, hierarchical_map = False, full_bayes = False, embed = False, savefig = False):

    list_values = [true_values, results_measures]
    list_sigma = [[0]*3, sigma]
    list_labels = ["True value", "Measure"]

    incr = 0.09 #distance between the error bars
    if hierarchical_map: plot_hierarchical_plugin = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) #get mean and std for hierarchical plugin
    if full_bayes: plot_full_bayes = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/full_bayes.csv", index_col=0) # get mean and std for hierarchical full bayes
    if no_error: plot_no_error = pd.read_csv(pre_path + f"no_error/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) # get mean and std for no error
    if unif_error: plot_unif_error = pd.read_csv(pre_path + f"uniform_error/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) #get mean and std for uniform error
    elinewidth = 3 #error bar width
    markersize = 8 
    if embed:
        Ysimu_embed = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/predictions.csv", index_col=0) #get mean for embedded discrepancy
        Ystd_embed = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/std_dev.csv", index_col=0) #get std for embedded discrepancy
    
    x = np.arange(len(results_measures))
    ticks = [f"$x_{{{k+1}}}$" for k in x]

    fig, axes = plt.subplots(3, 1, figsize=(30, 13))  # 3 subplots 
    
    for i, ax in enumerate(axes, start=1):
        sum_increments = 0
        if index_calib == i: #if the output is the one observed
            ax.errorbar(x, (list_values[1][f"Y{i}"]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=list_sigma[1][i-1]/list_values[0][f"Y{i}"], fmt='o', color='blue', label=list_labels[1],elinewidth=elinewidth, markersize = markersize) #plot measures and variance noise
        
        ax.scatter(x, list_values[0][f"Y{i}"]-list_values[0][f"Y{i}"], marker='x', color='blue', label=list_labels[0], s=120,linewidths=4) #plot true values
        if no_error: 
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_no_error.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_no_error.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='green', label='No error',elinewidth=elinewidth, markersize = markersize)  #plot no_error
        if unif_error: 
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_unif_error.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_unif_error.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='red', label='Uniform error',elinewidth=elinewidth,markersize = markersize) #plot uniform error
        if hierarchical_map:
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_hierarchical_plugin.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_hierarchical_plugin.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='purple', label="Hierarchical \n     MAP",elinewidth=elinewidth,markersize = markersize) #plot hierarchical map
        if full_bayes:
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_full_bayes.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_full_bayes.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='magenta', label="Hierarchical \n full Bayes",elinewidth=elinewidth,markersize = markersize) #plot full bayes
        if embed: 
            sum_increments += 1
            ax.errorbar(x + 5*incr, (Ysimu_embed.iloc[:, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=Ystd_embed.iloc[:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='orange', label="Embedded \ndiscrepancy",elinewidth=elinewidth,markersize = markersize) #plot embedded discrepancy
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(f"Prediction of {variable_names[i-1]} from measures of {variable_names[index_calib-1]}", fontsize=42)
        
        if i == len(axes): #x ticks on the last subplots
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, fontsize=30)
        else:
            ax.set_xticks([])  # Remove x-ticks for the first two subplots
        
        ax.tick_params(axis='y', labelsize=20)
    
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30, bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout()
    if savefig: 
        if(os.path.isdir(pre_path + "plots")==False):  
            os.mkdir(pre_path + "plots")
        plt.savefig(pre_path + f"plots/plot_pred_{index_calib}.jpg",bbox_inches='tight',format='jpg')    
    plt.show()
    
def compare_errors(index_calib,pre_path,no_error = False, unif_error = False, hierarchical_map = False, full_bayes = False, embed = False):
    columns = []
    res = np.empty((3,0))
    res_intervals = np.empty((3,0))
    if no_error: 
        errors_noerror = pd.read_csv(pre_path + f"no_error/calib_{index_calib}/errors_map.csv", index_col = 0).values.transpose() #get error no error
        intervals_noerror = pd.read_csv(pre_path + f"no_error/calib_{index_calib}/intervals_map.csv", index_col = 0).values #get interval levels no error
        res = np.concatenate([res, errors_noerror.reshape(-1,1)], axis = 1)
        res_intervals = np.concatenate([res_intervals,np.apply_along_axis(lambda x: sorted(x)[-2], 0, intervals_noerror).reshape(-1,1)], axis=1)
        columns.append("No error")
    if unif_error: 
        errors_allp = pd.read_csv(pre_path + f"uniform_error/calib_{index_calib}/errors_map.csv", index_col = 0).values.transpose() #get error uniform error
        intervals_allp = pd.read_csv(pre_path + f"uniform_error/calib_{index_calib}/intervals_map.csv", index_col = 0).values #get interval levels uniform error
        res = np.concatenate([res, errors_allp.reshape(-1,1)], axis = 1)
        res_intervals = np.concatenate([res_intervals,np.apply_along_axis(lambda x: sorted(x)[-2], 0, intervals_allp).reshape(-1,1)], axis=1)
        columns.append("  Uniform \n errors")
    if hierarchical_map:
        errors_MAP = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/errors_map.csv", index_col = 0).values.transpose() #get error hierarchical MAP
        intervals_MAP = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/intervals_map.csv", index_col = 0).values #get interval levels hierarchical MAP
        res = np.concatenate([res, errors_MAP.reshape(-1,1)], axis = 1)
        res_intervals = np.concatenate([res_intervals,np.apply_along_axis(lambda x: sorted(x)[-2], 0, intervals_MAP).reshape(-1,1)], axis=1)
        columns.append("Hierarchical \n MAP")
    if full_bayes: 
        errors_bayes = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/errors_bayes.csv", index_col = 0).values.transpose() #get error full bayes
        intervals_bayes = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/intervals_bayes.csv", index_col = 0).values #get interval levels full bayes
        res = np.concatenate([res, errors_bayes.reshape(-1,1)], axis = 1)
        res_intervals = np.concatenate([res_intervals,np.apply_along_axis(lambda x: sorted(x)[-2], 0, intervals_bayes).reshape(-1,1)], axis=1)
        columns.append("  Hierarchical \n full Bayes")
    if embed: 
        errors_embed = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/error_pred.csv", index_col = 0).values.transpose() #get error embedded discrepancy
        intervals_embed = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/interv_errors.csv", index_col = 0).values #get error embedded discrepancy
        res = np.concatenate([res, errors_embed.reshape(-1,1)], axis = 1)
        res_intervals = np.concatenate([res_intervals,np.apply_along_axis(lambda x: sorted(x)[-2], 0, intervals_embed).reshape(-1,1)], axis=1)
        columns.append(" Embedded \n discrepancy")


    return pd.DataFrame(res), pd.DataFrame(res_intervals), columns


def plot_errors(index_calib, pre_path, no_error = False, unif_error = False, hierarchical_map = False, full_bayes = False, embed = False, savefig = False):
    errors = compare_errors(index_calib=index_calib, pre_path=pre_path,  no_error = no_error, unif_error = unif_error, hierarchical_map = hierarchical_map, full_bayes = full_bayes, embed = embed) #Get RMSRE and p^0.9_M,N
    names = errors[2]
    errors = errors[0] * 100, errors[1] * 100 #convert to %
    x = np.arange(len(names))
    bar_width1 = 23 / 100
    bar_width2 = 18 / 100
    alph = 0.7
    fonttext = 35
    loc_bar = [-2 / 3 * bar_width1, 2 / 3 * bar_width2]
    loc_bar = loc_bar + [loc_bar[1] + bar_width2, loc_bar[1] + 2 * bar_width2]

    fig, axes = plt.subplots(3, 1, figsize=(37, 13), sharex=True, sharey=True)  # 3 rows, 1 column

    patches = [
        mpatches.Patch(color='#1f77b4', label='RMSRE (%)', alpha=alph),
        mpatches.Patch(color='green', label=r'$\hat{p}^{0.9}$ (%)', alpha=alph),
    ] # legend

    for idx, ax1 in enumerate(axes, start=1):
        ax1.bar(x + loc_bar[0], errors[0].iloc[idx-1, :].values, width=bar_width1, alpha=alph) #bars for RMSRE
        
        ax2 = ax1.twinx() #twin y-axis

        ax2.bar(x + loc_bar[1], errors[1].iloc[idx-1, :].values, color="green", width=bar_width1, alpha=alph) #bars for p^0.9

        for kk in range(len(x)):
            ax1.text(x[kk] + loc_bar[0], errors[0].iloc[idx-1, kk] * 0.95, "{:.1f}".format(errors[0].iloc[idx-1, kk]), ha='center', va="top", fontsize=fonttext, fontweight='bold') #text value of rmsre
            ax2.text(x[kk] + loc_bar[1], errors[1].iloc[idx-1, kk] * 0.95, str(round(errors[1].iloc[idx-1, kk])), ha='center', va="top", fontsize=fonttext, fontweight='bold') #text value of p^0.9

        ax1.tick_params(axis='x', labelsize=30)
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)

        ax1.locator_params(axis='y', nbins=3)
        ax2.locator_params(axis='y', nbins=3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(names)

        ax1.set_title(f"Prediction error of {variable_names[idx-1]} from measures of {variable_names[index_calib-1]}", fontsize=42)

    fig.legend(handles=patches, loc='center right', fontsize=40, bbox_to_anchor=(1.1, 0.5))

    fig.text(0.04, 0.5, 'RMSRE (%)', va='center', rotation='vertical', fontsize=38)
    fig.text(0.94, 0.5, 'Prediction interval level (%)', va='center', rotation='vertical', fontsize=38)

    plt.tight_layout(rect=[0.05, 0, 0.93, 1])  # Adjust layout to make room for y labels

    if savefig: 
        if(os.path.isdir(pre_path + "plots")==False):  
            os.mkdir(pre_path + "plots")
        plt.savefig(pre_path + f"plots/plot_err_{index_calib}.jpg",bbox_inches='tight', format = "jpg")
    
    plt.show()

