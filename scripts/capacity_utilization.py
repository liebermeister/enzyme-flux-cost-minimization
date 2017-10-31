#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:45:55 2017

@author: eladn
"""
from prepare_data import get_df_from_pareto_zipfile, \
                         read_pareto_zipfile
import definitions as D
from sensitivity_analysis import Sensitivity
import os
import numpy as np
import seaborn as sns
import pandas as pd           
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    # Make CSV files which summarize the most important results
    fig_name_and_titles = [('standard',  'glucose = %g mM, O$_2$ = %g mM' % 
                            (D.STD_CONC['glucoseExt'], D.STD_CONC['oxygen'])),
                           ('anaerobic', 'glucose = %g mM, no O$_2$' % 
                            (D.STD_CONC['glucoseExt']))]
    for fig_name, fig_title in fig_name_and_titles:

        # write the results for all EFMs to a CSV file
        zip_fnames, regex = D.DATA_FILES[fig_name]
        df = read_pareto_zipfile(zip_fnames[0])
        df = df[[D.YIELD_L, D.TOT_FLUX_SA_L, D.TOT_ENZYME_L, D.GROWTH_RATE_L]]
        df.round(3).to_csv(os.path.join(D.OUTPUT_DIR, '%s.csv' % fig_name))

        fname = os.path.join(D.OUTPUT_DIR, '%s_capacity_utilization.pdf' % fig_name)
        with PdfPages(fname) as pdf:
            
            # read the raw files again, now including all kinetic parameters
            # and individual enzyme abundances at the ECM optimum
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
            params.rename(columns={'kcat': 'kcat_fwd [1/s]'}, inplace=True)
            params['kcat_rev [1/s]'] = \
                params['kcat_fwd [1/s]'] * params['prod(Km)'] / params['Keq']
    
            # enzyme abundances in the result files are given in moles, and 
            # are the optimal amounts that enable the catalysis of the reaction
            # according to the rates in the rates. Multiplying the abundance
            # by the kcat values would give us the maximal capacity, which is 
            # higher than the actual rate (given in "rates")
            rates = rates.reset_index().melt(
                    id_vars='efm', var_name='reaction', value_name='rate [mM/s]')
            enzymes = enzymes.reset_index().melt(
                    id_vars='efm', var_name='reaction', value_name='enzyme [mM]')

            caputil_df = pd.merge(rates, enzymes, on=['efm', 'reaction'])
            
            # drop the cases where the enzyme levels were 0
            caputil_df = caputil_df[caputil_df['enzyme [mM]'] > 0]

            caputil_df['kapp [1/s]'] = \
                caputil_df['rate [mM/s]'] / caputil_df['enzyme [mM]']
            
            # to calculate the capacity usage,
            # we need to divide each kapp by the kcat, which 
            # is tricky, because it depends on the flux direction.
            caputil_df = caputil_df.join(
                params[['kcat_fwd [1/s]', 'kcat_rev [1/s]']], on='reaction')
            
            caputil_df['kcat [1/s]'] = caputil_df['kcat_fwd [1/s]']

            # for all cases where the flux is negative
            rev_idx = caputil_df[caputil_df['rate [mM/s]'] < 0].index
            caputil_df.loc[rev_idx, 'kcat [1/s]'] = caputil_df.loc[rev_idx, 'kcat_rev [1/s]']
            caputil_df.loc[rev_idx, 'kapp [1/s]'] = -caputil_df.loc[rev_idx, 'kapp [1/s]']
            
            caputil_df['capacity utilization'] = \
                caputil_df['kapp [1/s]'] / caputil_df['kcat [1/s]']

            caputil_exp = caputil_df[caputil_df['efm'] == 9999].set_index('reaction')
            caputil_df = caputil_df[caputil_df['efm'] != 9999]
            
            cap_util_median = caputil_df[['reaction', 'kcat [1/s]', 'capacity utilization']]
            cap_util_median = cap_util_median.groupby('reaction').median()
            cap_util_median['color'] = list(map(D.reaction_to_rgb, cap_util_median.index))
            order = list(cap_util_median.sort_values(by='capacity utilization').index)
    
            # TODO: add the cap. util. of the "exp" flux mode in a different color
            # and also the measured one for glucose media (get from Dan)
            fig1, ax = plt.subplots(1, 1, figsize=(15, 7))
            ax_box = sns.boxplot(x='reaction', y='capacity utilization',
                                 data=caputil_df,
                                 order=order, ax=ax)
            plt.xticks(rotation=90)
            boxes = ax_box.artists
            for i, r in enumerate(order):
                boxes[i].set_facecolor(D.reaction_to_rgb(r))
                if r in caputil_exp.index:
                    ax.plot(i, caputil_exp.at[r, 'capacity utilization'],
                            markerfacecolor=(1, 0, 0), marker='o')
                else:
                    ax.plot(i, -0.05, markerfacecolor=(1, 0, 0), marker='.')
    
            ax.set_title(fig_title)
            pdf.savefig(fig1)

            caputil_df['color'] = caputil_df['reaction'].apply(D.reaction_to_rgb)

            fig2, axs = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    
            axs[0].scatter(x=np.abs(caputil_df['rate [mM/s]']),
                           y=caputil_df['capacity utilization'],
                           c=caputil_df['color'], s=15, linewidth=0)
            axs[0].set_xscale('log')
            axs[0].set_xlim(1e-1, None)
            axs[0].set_xlabel('rate [mM/s]')
            axs[0].set_ylabel('capacity utilization')
            axs[0].set_title('cap. util. vs. flux')
            
            axs[1].scatter(x=cap_util_median['kcat [1/s]'],
                           y=cap_util_median['capacity utilization'],
                           c=cap_util_median['color'], s=50, linewidth=0)
            axs[1].set_xscale('log')
            axs[1].set_xlabel('kcat [1/s]')
            axs[1].set_title('median(cap.) vs. kcat')
            
            s = Sensitivity(fig_name)
            elast = s.efm_data_df[['efm', 'reaction', 'dlnmu/dlnk']]
            X3 = pd.merge(caputil_df, elast, on=['efm', 'reaction'])
            axs[2].scatter(x=-X3['dlnmu/dlnk'], y=X3['capacity utilization'],
                           c=X3['color'], s=15, linewidth=0)
            axs[2].set_xscale('log')
            axs[2].set_xlim(1e-4, None)
            axs[2].set_xlabel('-dlnmu/dlnk')
            axs[2].set_title('cap. util. vs. kcat sensitivity')
    
            fig2.tight_layout()
            pdf.savefig(fig2)
            