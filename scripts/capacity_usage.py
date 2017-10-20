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
import pandas as pd           
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Make CSV files which summarize the most important results
    fig_name_and_titles = [('standard',  'glucose = %g mM, O$_2$ = %g mM' % 
                            (D.STD_CONC['glucoseExt'], D.STD_CONC['oxygen'])),
                           ('anaerobic', 'glucose = %g mM, no O$_2$' % 
                            (D.STD_CONC['glucoseExt']))]
    for fig_name, fig_title in fig_name_and_titles:
        zip_fnames, regex = D.DATA_FILES[fig_name]
        df = read_pareto_zipfile(zip_fnames[0])
        df = df[[D.YIELD_L, D.TOT_FLUX_SA_L, D.TOT_ENZYME_L, D.GROWTH_RATE_L]]
        df.round(3).to_csv(os.path.join(D.OUTPUT_DIR, '%s.csv' % fig_name))
        
        rates, params, kms, enzymes = get_df_from_pareto_zipfile(zip_fnames[0])
        rates.round(3).to_csv(
            os.path.join(D.OUTPUT_DIR, '%s_rates.csv' % fig_name))
        enzymes.round(3).to_csv(
            os.path.join(D.OUTPUT_DIR, '%s_enzyme_abundance.csv' % fig_name))

        # calculate the reverse kcats for the reversible reactions
        kms['s_i * K_i'] = np.log(kms['Km']) * kms['coefficient']
        pi_km = kms.groupby(['reaction']).sum()['s_i * K_i']
        pi_km = pi_km.apply(np.exp)
        params['prod(Km)'] = pi_km
        params['kcat_rev'] = params['kcat'] * params['prod(Km)'] / params['Keq']

        # enzyme abundances in the result files are given in moles, and 
        # are the optimal amounts that enable the catalysis of the reaction
        # according to the rates in the rates. Multiplying the abundance
        # by the kcat values would give us the maximal capacity, which is 
        # higher than the actual rate (given in "rates")
        capacity_usage = rates / enzymes
        capacity_usage = capacity_usage.reset_index().melt(
                id_vars='efm', var_name='reaction', value_name='kapp')
        
        # drop the cases where the rate and enzyme levels were 0
        capacity_usage = capacity_usage[~pd.isnull(capacity_usage['kapp'])]

        # to calculate the capacity usage,
        # we need to divide each kapp by the kcat, which 
        # is tricky, because it depends on the flux direction.
        capacity_usage = capacity_usage.join(params[['kcat', 'kcat_rev']], on='reaction')
        
        capacity_usage['capacity usage'] = capacity_usage['kapp'] / capacity_usage['kcat']
        # override all cases where the kapp is negative:
        rev_idx = capacity_usage[capacity_usage['kapp'] < 0].index
        capacity_usage.loc[rev_idx, 'capacity usage'] = \
            -capacity_usage.loc[rev_idx, 'kapp'] / capacity_usage.loc[rev_idx, 'kcat_rev']
            
        order = capacity_usage.groupby('reaction').median()
        order = list(order.sort_values(by='capacity usage').index)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))
        g = sns.boxplot(x='reaction', y='capacity usage', data=capacity_usage,
                        order=order, ax=ax)
        plt.xticks(rotation=90)
        ax.set_title(fig_title)
        fig.savefig(
            os.path.join(D.OUTPUT_DIR, '%s_capacity_usage.pdf' % fig_name))

