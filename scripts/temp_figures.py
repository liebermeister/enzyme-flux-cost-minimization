#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:45:55 2017

@author: eladn
"""
from prepare_data import get_df_from_pareto_zipfile, \
                         read_pareto_zipfile
import definitions as D
import os
import numpy as np
import seaborn as sns                         
import matplotlib.pyplot as plt

if False:
    #%% Make CSV files which summarize the most important results
    for fig_name in ['standard', 'anaerobic']:
        zip_fnames, regex = D.DATA_FILES[fig_name]
        df = read_pareto_zipfile(zip_fnames[0])
        df = df[[D.YIELD_L, D.TOT_FLUX_SA_L, D.TOT_ENZYME_L, D.GROWTH_RATE_L]]
        df.round(3).to_csv(os.path.join(D.OUTPUT_DIR, '%s.csv' % fig_name))
        
        rates, params, enzymes = get_df_from_pareto_zipfile(zip_fnames[0])
        rates.round(3).to_csv(
            os.path.join(D.OUTPUT_DIR, '%s_rates.csv' % fig_name))
        enzymes.round(3).to_csv(
            os.path.join(D.OUTPUT_DIR, '%s_enzyme_abundance.csv' % fig_name))

        # enzyme abundances in the result files are given in moles, and 
        # are the optimal amounts that enable the catalysis of the reaction
        # according to the rates in the rates. Multiplying the abundance
        # by the kcat values would give us the maximal capacity, which is 
        # higher than the actual rate (given in "rates")
        capacity_usage = rates / (enzymes * params['kcat'])
        capacity_usage[capacity_usage < 0 ] = np.nan
        capacity_usage = capacity_usage.melt(var_name='reaction',
                                             value_name='capacity usage')
        
        capacity_usage = capacity_usage.dropna()
        order = list(capacity_usage.groupby('reaction').mean().sort_values(by='capacity usage').index)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))
        g = sns.boxplot(x='reaction', y='capacity usage', data=capacity_usage,
                        order=order, ax=ax)
        plt.xticks(rotation=90)
        ax.set_title(fig_name)
        fig.savefig(
            os.path.join(D.OUTPUT_DIR, '%s_capacity_usage.pdf' % fig_name))

