# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:38:03 2015

@author: noore

Estimate what the maximal theoretical growth rate per EFM would be if all
enzymes were saturated (i.e. the kcat was realized).
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import definitions as D
from phase_surface_plots import allocation_pie_chart
from prepare_data import get_concatenated_raw_data

figure_data = D.get_figure_data()

rcParams['font.size'] = 14.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['legend.fontsize'] = 'medium'
rcParams['axes.labelsize'] = 14.0
rcParams['axes.titlesize'] = 14.0
rcParams['xtick.labelsize'] = 12.0
rcParams['ytick.labelsize'] = 12.0

# %% Figure 2c
fig2c, ax2c = plt.subplots(1, 1, figsize=(5, 5))

data = figure_data['standard']
# remove oxygen-sensitive EFMs
data.loc[data[D.STRICTLY_ANAEROBIC_L], D.GROWTH_RATE_L] = 0
D.plot_basic_pareto(data, ax2c, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                    efm_dict=D.efm_dict,
                    facecolors=D.PARETO_NEUTRAL_COLOR, edgecolors='none')
ax2c.set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
ax2c.set_ylim(-1e-3, 1.15*data[D.GROWTH_RATE_L].max())
ax2c.set_title('glucose = 100 mM, O$_2$ = 3.7 mM')
fig2c.tight_layout()

fig2c.savefig(os.path.join(D.OUTPUT_DIR, 'Fig_web4.pdf'))

# %% histogram of all different EFM growth rates in a specific condition
fig5 = plt.figure(figsize=(5, 5))
ax5 = fig5.add_subplot(1, 1, 1)

efm = allocation_pie_chart(ax5, D.STD_CONC['glucoseExt'],
                           D.STD_CONC['oxygen'])
rates_df, full_df = get_concatenated_raw_data('sweep_glucose')

df = full_df[full_df['efm'] == efm]
v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * rates_df.at[efm, D.R_BIOMASS]

# make a new DataFrame where the index is the glucose concentration
# and the columns are the reactions and values are the costs.
absol = full_df[full_df['efm'] == efm].pivot(index=full_df.columns[1],
                                             columns='reaction',
                                             values='E_i')
fig5.savefig(os.path.join(D.OUTPUT_DIR, 'Fig_web3.pdf'))
