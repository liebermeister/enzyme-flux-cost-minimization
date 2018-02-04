"""
Created on Sat Dec  3 17:32:45 2016

@author: eladn
"""
import matplotlib.pyplot as plt
import definitions as D
from prepare_data import get_df_from_sweep_zipfile
from sensitivity_analysis import Sensitivity
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

#%%
GLU_CONC = ['10000.', '1000.', '100.', '10.', '1.', '0.1', '0.01', '0.001', '0.0001']
color_cycle = sns.color_palette(n_colors=len(GLU_CONC))
met = 'glucoseExt'
rxn = 'R1' # the glucose transporter
[zip_fname], regex = D.DATA_FILES['monod_glucose_aero']

figure_data = D.get_figure_data()

efm_data_dfs = {}
km_dfs = {}
for i, c in enumerate(GLU_CONC):
    s = Sensitivity(zip_fname, 'mext-%s-%s' % (met, c))
    efm_data_dfs[c] = s.efm_data_df.set_index(['efm', 'reaction'])
    km_dfs[c] = s.km_sensitivity_df.set_index(['efm', 'reaction', 'metabolite'])

#%% get the q as a function of c from the glucose sweep data
_, data_df = get_df_from_sweep_zipfile(zip_fname, regex)
q_monod_df = data_df.groupby(('efm', regex)).sum()['E_i'] / (D.BIOMASS_MW * D.SECONDS_IN_HOUR)
q_monod_df = q_monod_df.reset_index().pivot(regex, 'efm', 'E_i')

#%%
monod_list = []
for efm in D.efm_dict.keys():
    monod_curves = {}
    for i, c in enumerate(GLU_CONC):
        C0 = float(c)
        q0 = efm_data_dfs[c].at[(efm, rxn), 'q']
        dlnq_dlnC = -km_dfs[c].at[(efm, rxn, met), 'dlnq/dlnKm']
        Km = -C0 / (1.0 + 1.0 / dlnq_dlnC)
        qmin = q0 / (1.0 + Km/C0)
        color = color_cycle[i]
        monod_list.append((efm, C0, q0, dlnq_dlnC, Km, qmin, color))
monod_curve_df = pd.DataFrame(data=monod_list,
                              columns=['efm', 'C0', 'q0', 'dlnq_dlnC', 'Km', 'qmin', 'color'])
monod_curve_df = monod_curve_df.set_index(['efm', 'C0'])

#%%
# lnq as a function of lnC:
MONOD_FUNC = lambda lnC, a0, a1, a2 : a0 + np.log(1.0 + np.exp(-a1*lnC + a2))
p0 = (0.0, 1.0, 0.0)
LB = (-1e2, 0, -1e2)
UB = (1e2, 1, 1e2)

fitted_df = pd.DataFrame(index=D.efm_dict.keys(),
                         columns=['a0', 'a1', 'a2'],
                         dtype=float)
for efm in D.efm_dict.keys():
    try:
        popt, _ = curve_fit(MONOD_FUNC, np.log(q_monod_df.index), np.log(q_monod_df[efm]),
                            p0=p0,
                            bounds=(LB, UB),
                            method='trf')
        fitted_df.loc[efm, :] = popt
    except RuntimeError:
        print("cannot resolve Monod curve for EFM %d" % efm)
        fitted_df.loc[efm, :] = np.nan

#%%
log_glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                           np.log10(D.MAX_CONC['glucoseExt']), 200)

log_glu_grid_low = np.logspace(-4.5, -2, 200)

with PdfPages(os.path.join(D.OUTPUT_DIR, 'monod_approximation.pdf')) as pdf:
    for efm, (_, label) in D.efm_dict.items():
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(label)
        
        ax = axs[0, 0]
        ax.set_title('actual Monod curve')
        ax.plot(q_monod_df.index, q_monod_df[efm], '-',
                color='grey', alpha=0.7)
        
        for c in GLU_CONC:
            C0 = float(c)
            d = monod_curve_df.loc[(efm, C0), :]
            q = lambda c : d['qmin'] * (1.0 + d['Km']/c)
            ax.plot(log_glu_grid, list(map(q, log_glu_grid)),
                    '-', color=d['color'], alpha=0.7)
            ax.plot(C0, d['q0'], 'o', color=d['color'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('External glucose [mM]')
        ax.set_ylabel('$q$')
        ax.set_ylim(None, 100)


        ax = axs[0, 1]
        ax.set_title('fitted hyperbolic function')
        q_fit = lambda C : np.exp(MONOD_FUNC(np.log(C), *fitted_df.loc[efm, :]))
        ax.plot(log_glu_grid, list(map(q_fit, log_glu_grid)),
                '-', color='grey', alpha=0.7)

        a0, a1, a2 = fitted_df.loc[efm, :]
        q_low = lambda C : np.exp(a0 - a1*np.log(C) + a2)
        ax.plot(log_glu_grid_low, list(map(q_low, log_glu_grid_low)),
                '--', color='k', alpha=0.7)
        q_high = lambda C : np.exp(a0)
        ax.plot(log_glu_grid, list(map(q_high, log_glu_grid)),
                '--', color='k', alpha=0.7)
        
        for c in GLU_CONC:
            C0 = float(c)
            q0 = monod_curve_df.at[(efm, C0), 'q0']
            ax.plot(C0, q0, 'ro')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('External glucose [mM]')
        ax.set_ylabel('$q$')
        ax.set_ylim(1e-2, 100)

        ax = axs[1, 0]
        ax.set_title('edge-extrapolated hyperbolic function')
        
        c_low = monod_curve_df.index.levels[1].min()
        q_low = monod_curve_df.loc[(efm, c_low), 'q0']
        s_low = monod_curve_df.loc[(efm, c_low), 'dlnq_dlnC']

        c_high = monod_curve_df.index.levels[1].max()
        q_high = monod_curve_df.loc[(efm, c_high), 'q0']
        s_high = monod_curve_df.loc[(efm, c_high), 'dlnq_dlnC']
        
        a0 = np.log(q_high)
        a1 = -s_low / (1.0 - q_high/q_low)
        a2 = np.log(q_low/q_high) + a1*np.log(c_low)
        q_fit = lambda C : np.exp(MONOD_FUNC(np.log(C), a0, a1, a2))
        ax.plot(log_glu_grid, list(map(q_fit, log_glu_grid)),
                '-', color='grey', alpha=0.7)

        for c in GLU_CONC:
            C0 = float(c)
            q0 = monod_curve_df.at[(efm, C0), 'q0']
            ax.plot(C0, q0, 'ro')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('External glucose [mM]')
        ax.set_ylabel('$q$')
        ax.set_ylim(1e-2, 100)
        fig.tight_layout(h_pad=1, w_pad=1)
        
        pdf.savefig(fig)
