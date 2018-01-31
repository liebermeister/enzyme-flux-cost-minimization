"""
Created on Sat Dec  3 17:32:45 2016

@author: eladn
"""
import pandas as pd
import matplotlib.pyplot as plt
import definitions as D
from monod_surface import get_glucose_sweep_df, PREFIX, REGEX
import os
import zipfile
from sensitivity_analysis import Sensitivity
import numpy as np
from prepare_data import get_df_from_sweep_zipfile
import sys

#%%
figure_data = D.get_figure_data()
efm = 1565 # max-gr
rxn = 'R1' # the glucose transporter
met = 'glucoseExt'

#%%
monod_df = figure_data['monod_glucose_aero'].loc[efm, :].reset_index()
monod_df.columns = [D.GLU_COL, D.GROWTH_RATE_L]

#%%
#s = Sensitivity(D.DATA_FILES['monod_glucose_aero'][0][0],
#                'mext-glucoseExt-100.')
#conc0 = 100.

s = Sensitivity(D.DATA_FILES['monod_glucose_aero'][0][0],
                'mext-glucoseExt-0.1')
conc0 = 0.1

#%%
q0 = s.efm_data_df[(s.efm_data_df.efm == efm) &
                   (s.efm_data_df.reaction == rxn)]['q'].values[0]

# assuming a linear function for the conversion of cost to growth rate (i.e.
# not the standard way we decpict growth rate in the paper)
dlnq_dlnC = -s.km_sensitivity_df[(s.km_sensitivity_df.efm == efm) & 
                                 (s.km_sensitivity_df.reaction == rxn) &
                                 (s.km_sensitivity_df.metabolite == met)]['dlnq/dlnKm'].values[0]

Kmonod = -conc0 / (1.0 + 1.0 / dlnq_dlnC)
qmin = q0 / (1 + Kmonod/conc0)
mu_100 = lambda C : D.GR_FUNCTION(C / (qmin * (C + Kmonod)))


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                       np.log10(D.MAX_CONC['glucoseExt']), 200)
ax.set_xscale('log')
ax.plot(glu_grid, list(map(mu_100, glu_grid)), 'r:')
ax.plot(monod_df[D.GLU_COL], monod_df[D.GROWTH_RATE_L], 'g-')
ax.plot(conc0, D.GR_FUNCTION(1/q0), 'o')


#%%
#def plot_1D_sweep(interpolated_df, ax, color_func=None):
#    best_df = pd.DataFrame(index=interpolated_df.index,
#                           columns=[D.GROWTH_RATE_L, 'best_efm', 'hexcolor'])
#    best_df[D.GROWTH_RATE_L] = interpolated_df.max(axis=1)
#    best_df['best_efm'] = interpolated_df.idxmax(axis=1)
#    
#    best_efms = sorted(best_df['best_efm'].unique())
#    
#    if color_func is None:
#        color_dict = dict(zip(best_efms, D.cycle_colors(len(best_efms),
#                                                        h0=0.02, s=1)))
#        color_func = color_dict.get
#        
#    best_df['hexcolor'] = best_df['best_efm'].apply(color_func)
#    ax.plot(interpolated_df.index, interpolated_df, '-',
#             linewidth=1, alpha=0.2, color=(0.5, 0.5, 0.8))
#    #ax.set_xscale('log')
#    ax.set_xlim(0.6e-4, 0.2)
#    ax.set_ylim(1e-3, 0.86)
#    ax.set_xlabel(D.GLU_COL)
#    ax.set_ylabel(D.GROWTH_RATE_L)
#    
##fig, ax = plt.subplots(1, 1, figsize=(10, 10))
##interpolated_df = get_glucose_sweep_df(oxygen_conc=D.STD_CONC['oxygen'])
##plot_1D_sweep(interpolated_df, ax, D.efm_to_hex)
#
##fig.tight_layout()
##D.savefig(figS25, 'S25')

#s1 = Sensitivity.from_figure_name('standard')

#%% plot the Monod curve and approximation hyperbolic lower bounds
#   for a specific EFM



