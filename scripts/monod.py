# -*- coding: utf-8 -*-
"""
Created on Wed Oct 1 2015

@author: noore
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
import definitions as D
import pandas as pd


#LOW_GLUCOSE = D.LOW_CONC['glucoseExt']
LOW_GLUCOSE = 1e-3 # in mM, i.e. 1 uM

MAX_GROWTH_RATE_L = 'max growth rate [h$^{-1}$]'
GROWTH_RATE_LOW_GLU = 'growth rate at\n%g $\mu$M glucose [h$^{-1}$]' % (1e3*LOW_GLUCOSE)
MONOD_COEFF_L = 'Monod coefficient [mM glucose]'
INV_MONOD_COEFF_L = 'inverse of Monod coeff.\n[mM$^{-1}$]'
MAX_GR_OVER_KM_L = 'max. growth rate / $K_{Monod}$ \n[h$^{-1}$ mM$^{-1}$]'
HILL_COEFF_L = 'Hill coefficitent'

MONOD_FUNC = lambda x, gr_max, K_M, h : gr_max / (1 + (K_M/x)**h); p0 = (0.07, 1.0, 1.0)

def calculate_monod_parameters(figure_data):
    aerobic_data_df = figure_data['standard']
    aerobic_sweep_data_df = figure_data['monod_glucose_aero']
    anaerobic_data_df = figure_data['anaerobic'].drop(9999)
    anaerobic_sweep_data_df = figure_data['monod_glucose_anae'].drop(9999)

    aerobic_sweep_data_df = aerobic_sweep_data_df.transpose().fillna(0)
    anaerobic_sweep_data_df = anaerobic_sweep_data_df.transpose().fillna(0)

    plot_data = [('aerobic conditions', aerobic_sweep_data_df, aerobic_data_df),
                 ('anaerobic conditions', anaerobic_sweep_data_df, anaerobic_data_df)]
    
    monod_dfs = []
    for title, sweep_df, data_df in plot_data:
        monod_df = pd.DataFrame(index=sweep_df.columns,
                                columns=[MAX_GROWTH_RATE_L, MONOD_COEFF_L, HILL_COEFF_L],
                                dtype=float)
        for efm in monod_df.index:
            try:
                popt, _ = curve_fit(MONOD_FUNC, sweep_df.index, sweep_df[efm],
                                    p0=p0, method='trf')
                monod_df.loc[efm, :] = popt
            except RuntimeError:
                print("cannot resolve Monod curve for EFM %d" % efm)
                monod_df.loc[efm, :] = np.nan

        # get fig3 data for plotting the other features
        monod_df = monod_df.join(data_df)
        monod_df[INV_MONOD_COEFF_L] = 1.0/monod_df[MONOD_COEFF_L]
        monod_df[MAX_GR_OVER_KM_L] = monod_df[MAX_GROWTH_RATE_L] * monod_df[INV_MONOD_COEFF_L]

        # calculate the value of the growth rate using the Monod curve
        # for LOW_GLUCOSE

        monod_df[GROWTH_RATE_LOW_GLU] = 0
        for j in monod_df.index:
            monod_df.loc[j, GROWTH_RATE_LOW_GLU] = MONOD_FUNC(LOW_GLUCOSE,
                       monod_df.at[j, MAX_GROWTH_RATE_L],
                       monod_df.at[j, MONOD_COEFF_L],
                       monod_df.at[j, HILL_COEFF_L])
            
        monod_dfs.append((title, monod_df))
    
    return monod_dfs
        
def plot_monod_figure(monod_dfs, y_var=MAX_GROWTH_RATE_L):
    fig = plt.figure(figsize=(15, 14))
    gs1 = gridspec.GridSpec(2, 4, left=0.05, right=0.95, bottom=0.55, top=0.97)
    gs2 = gridspec.GridSpec(2, 4, left=0.05, right=0.95, bottom=0.06, top=0.45)

    axs = []
    for i in range(2):
        for j in range(4):
            axs.append(plt.subplot(gs1[i, j]))
    for i in range(2):
        for j in range(4):
            axs.append(plt.subplot(gs2[i, j]))

    for i, ax in enumerate(axs):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.95),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)

    for i, (title, monod_df) in enumerate(monod_dfs):
        xaxis_data = [(INV_MONOD_COEFF_L, (1, 2500), 'log'),
                      (GROWTH_RATE_LOW_GLU, (0.001, 0.2), 'linear')]
        for j, (x_var, xlim, xscale) in enumerate(xaxis_data):
            ax_row = axs[4*i + 8*j : 4*i + 8*j + 4]
            
            ax = ax_row[0]
            x = monod_df[x_var]
            y = monod_df[y_var]
            CS = ax.scatter(x, y, s=12, marker='o',
                            facecolors=(0.85, 0.85, 0.85),
                            linewidth=0)
            for efm, (col, lab) in D.efm_dict.items():
                if efm in x.index:
                    ax.plot(x[efm], y[efm], markersize=5, marker='o',
                            color=col, label=None)
                    ax.annotate(lab, xy=(x[efm], y[efm]),
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', va='bottom', color=col)
    
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_xscale(xscale)
            ax.set_title('%s' % title, fontsize=16)
            ax.set_xlabel(x_var, fontsize=16)
            ax.set_ylabel(y_var, fontsize=16)
    
            plot_parameters = [
                {'c': D.OXYGEN_L, 'title': 'oxygen uptake'    ,
                 'ax': ax_row[1], 'vmin': 0,    'vmax': 0.8},
                {'c': D.YIELD_L,  'title': 'yield'            ,
                 'ax': ax_row[2], 'vmin': 0,    'vmax': 30},
                {'c': D.ACE_L,    'title': 'acetate secretion',
                 'ax': ax_row[3], 'vmin': 0,    'vmax': 0.6}
                ]
    
            for d in plot_parameters:
                x = monod_df[x_var]
                y = monod_df[y_var]
                c = monod_df[d['c']]
                CS = d['ax'].scatter(x, y, s=12, c=c, marker='o',
                                     linewidth=0, cmap='copper_r',
                                     vmin=d['vmin'], vmax=d['vmax'])
                cbar = plt.colorbar(CS, ax=d['ax'])
                cbar.set_label(d['c'], fontsize=12)
                d['ax'].set_title(d['title'], fontsize=16)
                if i % 2 == 1:
                    d['ax'].set_xlabel(x_var, fontsize=16)
                d['ax'].set_xlim(xlim[0], xlim[1])
                d['ax'].set_xscale(xscale)

    for i in range(16):
        axs[i].set_yscale('linear')
        axs[i].set_ylim(0, 0.85)
        if i % 8 == 0:
            axs[i].get_xaxis().set_visible(False)
        if i % 4 > 0:
            axs[i].get_yaxis().set_visible(False)

    return fig

if __name__ == '__main__':
    figure_data = D.get_figure_data()
    monod_dfs = calculate_monod_parameters(figure_data)
    figS17 = plot_monod_figure(monod_dfs)
    D.savefig(figS17, 'S17')

    #%%
    from pandas import ExcelWriter
    writer = ExcelWriter(os.path.join(D.OUTPUT_DIR, 'monod_params.xls'))
    for title, monod_df in monod_dfs:
        monod_df.to_excel(writer, title)
    writer.save()