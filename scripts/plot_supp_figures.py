# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:32:45 2016

@author: eladn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D # NOTE!!! keep this for the 3D plots
import zipfile
import definitions as D
from prepare_data import get_concatenated_raw_data, get_df_from_pareto_zipfile
from sensitivity_analysis import Sensitivity
from phase_surface_plots import plot_surface, \
    plot_surface_diff, \
    plot_phase_plots, \
    plot_conc_versus_uptake_figure, \
    plot_glucose_dual_pareto, \
    plot_growth_rate_hist, \
    interpolate_single_condition, \
    plot_glucose_sweep, \
    SweepInterpolator, \
    get_glucose_sweep_df, \
    get_anaerobic_glucose_sweep_df
from monod import plot_monod_figure
from epistasis import Epistasis
from tsne import plot_tsne_figure
import tempfile
import shutil
import pareto_sampling
import seaborn as sns

figure_data = D.get_figure_data()

if __name__ == '__main__':
#if False:
    # %% Figure S1 - same as 3c, but compared to the biomass rate
    #    instead of growth rate
    figS1, axS1 = plt.subplots(1, 2, figsize=(9, 4.5))

    data = figure_data['standard']
    # remove oxygen-sensitive EFMs
    data.loc[data[D.STRICTLY_ANAEROBIC_L], D.GROWTH_RATE_L] = 0
    D.plot_basic_pareto(data, axS1[0], x=D.YIELD_L, y=D.BIOMASS_PROD_PER_ENZ_L,
                        facecolors=D.PARETO_NEUTRAL_COLOR, edgecolors='none')
    axS1[0].set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
    axS1[0].set_ylim(-1e-3, 1.15*data[D.BIOMASS_PROD_PER_ENZ_L].max())
    axS1[0].set_title('glucose = 100 mM, O$_2$ = 3.7 mM')
    axS1[0].annotate('a', xy=(0.02, 0.98),
                     xycoords='axes fraction', ha='left', va='top',
                     size=20)
    for y in range(0, 14, 2):
        axS1[0].plot([-1e-3, 1.1*data[D.YIELD_L].max()], [y, y], 'k-',
                     alpha=0.2)

    D.plot_basic_pareto(data, axS1[1], x=D.YIELD_L, y=D.GROWTH_RATE_L,
                        facecolors=D.PARETO_NEUTRAL_COLOR, edgecolors='none')
    axS1[1].set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
    axS1[1].set_ylim(-1e-3, 1.15*data[D.GROWTH_RATE_L].max())
    axS1[1].set_title('glucose = 100 mM, O$_2$ = 3.7 mM')
    axS1[1].annotate('b', xy=(0.02, 0.98),
                     xycoords='axes fraction', ha='left', va='top',
                     size=20)

    for y in map(D.GR_FUNCTION, range(0, 18, 2)):
        axS1[1].plot([-1e-3, 1.1*data[D.YIELD_L].max()], [y, y], 'k-',
                     alpha=0.2)

    figS1.tight_layout()
    D.savefig(figS1, 'S1')

    # %%
    # SI Figure 3: comparing sum of flux/SA to the total enzyme cost
    figS3, axS3 = plt.subplots(1, 1, figsize=(5, 5))
    data = figure_data['standard']
    inds_to_remove = data.isnull().any(axis=1) | data[D.STRICTLY_ANAEROBIC_L]
    data = data.loc[~inds_to_remove, :]
    D.plot_basic_pareto(data, x=D.TOT_FLUX_SA_L, y=D.TOT_ENZYME_L,
                        ax=axS3, edgecolors='none',
                        facecolors=D.PARETO_NEUTRAL_COLOR)
    axS3.set_xscale('log')
    axS3.set_yscale('log')
    minval, maxval = (1e-2, 40)

    # draw the x=y line
    axS3.plot([minval, maxval], [minval, maxval], '-',
              color='k', linewidth=1)

    # draw the two bounding diagonal lines (which bound the function x-y)
    # from above and below
    min_ratio = (data[D.TOT_ENZYME_L] / data[D.TOT_FLUX_SA_L]).min()
    max_ratio = (data[D.TOT_ENZYME_L] / data[D.TOT_FLUX_SA_L]).max()
    
    color_min = (1, 0, 0)
    color_max = (0, 0, 1)
    axS3.plot([minval, maxval/min_ratio], [minval*min_ratio, maxval], '-',
              color=color_min, linewidth=1)
    axS3.annotate(xy=(10, 10*min_ratio),
                  s='y = %.1f x' % min_ratio,
                  xycoords='data', xytext=(10, 3),
                  va='top', ha='center', color=color_min,
                  arrowprops=dict(color=color_min,
                                  shrink=0.02, width=1, headwidth=3))
    axS3.plot([minval, maxval/max_ratio], [minval*max_ratio, maxval], '-',
              color=color_max, linewidth=1, alpha=1)
    axS3.annotate(xy=(20/max_ratio, 20),
                  s='y = %.1f x' % max_ratio,
                  xycoords='data', xytext=(1, 20),
                  va='center', ha='right', color=color_max,
                  arrowprops=dict(color=color_max,
                                  shrink=0.02, width=1, headwidth=3))

    # mark the two extreme points (the ones with the minimal x and minimal
    # y values)
    min_x = data[D.TOT_FLUX_SA_L].min()
    min_y = data[D.TOT_ENZYME_L].min()
    for efm, row in data.iterrows():
        x = row[D.TOT_FLUX_SA_L]
        y = row[D.TOT_ENZYME_L]
        if x == min_x:
            axS3.annotate(xy=(x, y), s='min. ideal\ncost = %.3f [h$^{-1}$]' % min_x,
                          xycoords='data', fontsize=12, ha='center',
                          xytext=(x*1.4, y*20), rotation=0,
                          arrowprops=dict(color='black',
                          shrink=0.02, width=1, headwidth=3))
        if y == min_y:
            axS3.annotate(xy=(x, y), s='min. actual\ncost = %.3f [h$^{-1}$]' % min_y,
                          xycoords='data', fontsize=12, va='center',
                          xytext=(x*5, y), rotation=0,
                          arrowprops=dict(color='black',
                          shrink=0.02, width=1, headwidth=3))

    axS3.set_xlim(minval, maxval)
    axS3.set_ylim(minval, maxval)
    axS3.set_xlabel(r'ideal cost [gr enz / gr dw h$^{-1}$]')
    axS3.set_ylabel(r'actual cost [gr enz / gr dw h$^{-1}$]')
    figS3.tight_layout()
    D.savefig(figS3, 'S3')

    # %% SI Figure 5 - sweep for the kcat values of biomass reaction
    figS5, axS5 = plt.subplots(1, 1, figsize=(5, 5))
    D.plot_sweep(figure_data['sweep_kcat_r70'],
                 r'$k_{cat}$ of biomass reaction [s$^{-1}$]',
                 efm_dict=D.efm_dict, ax=axS5, legend_loc='lower center')
    axS5.set_xscale('log')
    maxy = axS5.axes.yaxis.get_view_interval()[1]
    axS5.plot([100, 100], [0.0, maxy], '--', color='grey', linewidth=1)
    axS5.text(100, maxy, r'default $k_{cat}$',
              va='bottom', ha='center', color='grey')
    D.savefig(figS5, 'S5')

    # %% SI Figure 6 - t-SNE projections
    figS6 = plot_tsne_figure(figure_data)
    D.savefig(figS6, 'S6')

    # %% SI Figure 7
    # make bar plots for each reaction, counting how many EFMs it participates
    figS7, axS7 = plt.subplots(2, 2, figsize=(15, 12))

    for i, ax in enumerate(axS7.flat):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)

    rates1_df, _, _, _ = get_concatenated_raw_data('standard')
    rates2_df, _, _, _ = get_concatenated_raw_data('anaerobic')
    rates_df = pd.concat([rates1_df, rates2_df]).drop_duplicates()

    reaction_counts = 100 * (rates_df.abs() > 1e-8).sum(0) / rates_df.shape[0]

    plt.subplots_adjust(hspace=0.3)
    reaction_counts.sort_values(inplace=True)
    reaction_counts.plot(kind='bar', ax=axS7[0, 0], color=D.BAR_COLOR, linewidth=0)
    axS7[0, 0].set_ylim(0, 100)
    axS7[0, 0].set_ylabel('\% of EFMs using this reaction')
    axS7[0, 0].set_xticklabels(map(D.GET_REACTION_NAME, reaction_counts.index))

    rates_df = rates_df.drop(9999)  # remove "exp" which is the last index
    efm_counts = (rates_df.abs() > 1e-8).sum(1)
    efm_counts.hist(bins=np.arange(20, 40)-0.5, ax=axS7[0, 1],
                    color=D.BAR_COLOR, rwidth=0.4)
    axS7[0, 1].set_xlabel('no. of active reactions')
    axS7[0, 1].set_ylabel('no. of EFMs')
    axS7[0, 1].set_xlim(22, 36)

    # Figure that calculates the correlation between each EFM and
    # the "experimental" flow, and overlays that information on the
    # standard "Pareto" plot

    data = figure_data['standard'].copy()
    data.loc[data[D.STRICTLY_ANAEROBIC_L], D.GROWTH_RATE_L] = 0

    CORR_FLUX_L = 'Flux Spearman correlation'
    CORR_ENZ_L = 'Enzyme Spearman correlation'

    # read the measured fluxes
    exp_flux_df = D.get_projected_exp_fluxes()
    # remove the exchange reactions (xchg_*)
    exp_flux_df = exp_flux_df.loc[exp_flux_df.index.str.find('xchg') != 0, :]
    exp_flux_df.index = map(D.FIX_REACTION_ID, exp_flux_df.index)

    rates_df, params_df, km_df, enzyme_abundance_df = \
        get_concatenated_raw_data('standard')

    # calculate correlation coefficients between the enzyme abundances and
    # the measured abundances (from Schmidt et al. 2015, glucose batch)
    X = enzyme_abundance_df.transpose()

    # in order to convert the enzyme abundances to realistic values, we need
    # to scale by a factor of 0.004 (see SI text, section S2.5)
    X *= 0.004

    y = map(D.PROTEOME_DICT.get, enzyme_abundance_df.columns)
    X['measured'] = pd.Series(index=enzyme_abundance_df.columns, data=y)
    X_pred = X.iloc[:, 0:-1].as_matrix()
    X_meas = X.iloc[:, -1].as_matrix()

    data[CORR_FLUX_L] = rates_df.transpose().corr('spearman').loc[9999]
    data[CORR_ENZ_L] = X.corr('spearman').loc['measured']

    # Pareto plot of correlation between predicted and measured fluxes
    axS7[1, 0].set_title('Match with measured fluxes')
    D.plot_basic_pareto(data, axS7[1, 0], x=D.YIELD_L, y=D.GROWTH_RATE_L,
                        c=CORR_FLUX_L, cmap='copper_r',
                        vmin=0, vmax=1, linewidth=0, s=30, edgecolor='k')

    # Pareto plot of correlation between predicted and measured enzyme levels
    axS7[1, 1].set_title('Match with measured enzyme abundance')
    D.plot_basic_pareto(data, axS7[1, 1], x=D.YIELD_L, y=D.GROWTH_RATE_L,
                        c=CORR_ENZ_L, cmap='copper_r',
                        vmin=0, vmax=1, linewidth=0, s=30, edgecolor='k')

    annot_color = (0.1, 0.1, 0.8)
    for ax in axS7[1, :]:
        ax.set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
        ax.set_ylim(-1e-3, 1.15*data[D.GROWTH_RATE_L].max())
        for efm in D.efm_dict.keys():
            xy = np.array(data.loc[efm, [D.YIELD_L, D.GROWTH_RATE_L]].tolist())
            xytext = xy.copy()
            xytext[0] = 0.1 * ax.get_xlim()[1] + 0.8 * xy[0]
            xytext[1] += 0.07
            ax.annotate(xy=xy, s=D.efm_dict[efm]['label'],
                        xycoords='data', xytext=xytext, ha='left', va='bottom',
                        color=annot_color, fontsize=16,
                        arrowprops=dict(facecolor=annot_color,
                        shrink=0.1, width=3, headwidth=6))

    figS7.tight_layout()
    D.savefig(figS7, 'S7')

    # %% SI Figure 8 - pareto plot with 4 alternative EFM features
    figS8, axS8 = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)

    plot_parameters = [{'title': 'succinate:fumarate cycling',
                        'c': D.SUC_FUM_CYCLE_L},
                       {'title': 'ammonia uptake', 'c': D.NH3_L},
                       {'title': 'ED pathway', 'c': D.ED_L},
                       {'title': 'pentose phosphate pathway', 'c': D.PPP_L},
                       {'title': 'upper glycolysis', 'c': D.UPPER_GLYCOLYSIS_L, 'cmap': 'coolwarm'},
                       {'title': 'pyruvate dehydrogenase', 'c': D.PDH_L}
                       ]
    data = figure_data['standard']
    for i, (ax, d) in enumerate(zip(list(axS8.flat), plot_parameters)):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)
        D.plot_basic_pareto(data, ax=ax,
                            x=D.YIELD_L, y=D.GROWTH_RATE_L,
                            c=d['c'], cmap=d.get('cmap', 'copper_r'),
                            linewidth=0, s=20)
        ax.set_title(d['title'])
        ax.set_xlim(-1e-3, 1.05*data[D.YIELD_L].max())
        ax.set_ylim(-1e-3, 1.05*data[D.GROWTH_RATE_L].max())

    figS8.tight_layout()
    D.savefig(figS8, 'S8')

    # %% SI Figure 9 - comparing yield to other EFM parameters
    figS9, axS9 = plt.subplots(2, 2, figsize=(7, 7))

    plot_parameters = [{'y': D.ACE_L, 'ymin': -0.001, 'ax': axS9[0, 0]},
                       {'y': D.OXYGEN_L, 'ymin': -0.001, 'ax': axS9[0, 1]},
                       {'y': D.N_REACTION_L, 'ymin': 20.0, 'ax': axS9[1, 0]},
                       {'y': D.TOT_FLUX_L, 'ymin': 0.0, 'ax': axS9[1, 1]}]

    data = pd.concat([figure_data['standard'], figure_data['anaerobic']])
    data = data.reset_index().groupby('EFM').first()
    data.fillna(0, inplace=True)
    for i, d in enumerate(plot_parameters):
        d['ax'].annotate(chr(ord('a')+i), xy=(0.04, 0.98),
                         xycoords='axes fraction', ha='left', va='top',
                         size=20)

        D.plot_basic_pareto(data, ax=d['ax'],
                            x=D.YIELD_L, y=d['y'], efm_dict=D.efm_dict,
                            edgecolors='none',
                            facecolors=(0.85, 0.85, 0.85),
                            show_efm_labels=True)
        d['ax'].set_xlim(-0.1, None)
        d['ax'].set_ylim(d['ymin'], None)
    figS9.tight_layout()
    D.savefig(figS9, 'S9')

    # %% SI figure 10 - histogram of all different EFM growth
    #    rates in a specific condition
    figS10, (axS10a, axS10b) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    plot_growth_rate_hist(ax=axS10a)
    plot_growth_rate_hist(oxygen=D.LOW_CONC['oxygen'], ax=axS10b)
    axS10b.set_ylabel('')

    figS10.tight_layout()
    D.savefig(figS10, 'S10')

    # %% SI Figure 11
    figS11, axS11 = plt.subplots(1, 1, figsize=(5, 5))
    plot_glucose_dual_pareto(figure_data['standard'], axS11,
                             draw_lines=False)
    axS11.set_xlim(-1e-3, None)
    axS11.set_ylim(-1e-3, None)
    D.savefig(figS11, 'S11')

    # %% SI Figure 12
    figS12 = plot_phase_plots(figure_data)
    figS12.tight_layout(pad=0.1)
    D.savefig(figS12, 'S12')

    # %% SI figure 13 - scatter 3D plot of the glucose uptake, oxygen uptake,
    #    growth rate
    figS13 = plot_conc_versus_uptake_figure(figure_data)
    figS13.tight_layout(w_pad=3.5, h_pad=2)
    D.savefig(figS13, 'S13')

    # %% SI figure 14 - scatter plots in different environmental conditions
    figS14, axS14 = plt.subplots(2, 2, figsize=(8, 8),
                                 sharex=True, sharey=True)


    params = [{'glucose': D.STD_CONC['glucoseExt'],
               'oxygen':  D.STD_CONC['oxygen'],
               'ax': axS14[0, 0]},
              {'glucose': D.STD_CONC['glucoseExt'],
               'oxygen': D.LOW_CONC['oxygen'],
               'ax': axS14[0, 1]},
              {'glucose': D.LOW_CONC['glucoseExt'],
               'oxygen': D.STD_CONC['oxygen'],
               'ax': axS14[1, 0]}]

    data = figure_data['standard']
    plot_list = [("require oxygen", (0.95, 0.7, 0.7),
                  ~data[D.STRICTLY_ANAEROBIC_L] & data[D.STRICTLY_AEROBIC_L]),
                 ("oxygen sensitive", (0.4, 0.7, 0.95),
                  data[D.STRICTLY_ANAEROBIC_L]  & ~data[D.STRICTLY_AEROBIC_L]),
                 ("facultative", (0.9, 0.5, 0.9),
                  ~data[D.STRICTLY_ANAEROBIC_L] & ~data[D.STRICTLY_AEROBIC_L])]

    x = D.YIELD_L
    y = D.GROWTH_RATE_L
    for d in params:
        ax = d['ax']
        ax.set_title('glucose = %g mM, O$_2$ = %g mM' %
                     (d['glucose'], d['oxygen']))
        gr = interpolate_single_condition(glucose=d['glucose'],
                                          oxygen=d['oxygen'])
        for label, color, efms in plot_list:
            xdata = data.loc[efms, x]
            ydata = gr[efms]
            ax.scatter(xdata, ydata, s=12, marker='o', alpha=1,
                       edgecolors='none', color=color,
                       label=label)
        for efm, (col, lab) in D.efm_dict.items():
            if efm in data.index:
                ax.plot(data.at[efm, x], gr[efm], markersize=5,
                        marker='o', color=col, label=None)
                ax.annotate(lab, xy=(data.at[efm, x], gr[efm]),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', va='bottom', color=col)

    # plot the anaerobic condition data
    ax = axS14[1, 1]
    ax.set_title('glucose = %g mM, no O$_2$' % D.STD_CONC['glucoseExt'])
    data = figure_data['anaerobic'].copy().drop(9999)
    plot_list = [("require oxygen", (0.95, 0.7, 0.7), []),
                 ("oxygen sensitive", (0.4, 0.7, 0.95),
                  data[D.STRICTLY_ANAEROBIC_L]  & ~data[D.STRICTLY_AEROBIC_L]),
                 ("facultative", (0.9, 0.5, 0.9),
                  ~data[D.STRICTLY_ANAEROBIC_L] & ~data[D.STRICTLY_AEROBIC_L])]
    for label, color, efms in plot_list:
        xdata = data.loc[efms, D.YIELD_L]
        ydata = data.loc[efms, D.GROWTH_RATE_L]
        ax.scatter(xdata, ydata, s=12, marker='o', alpha=1,
                   edgecolors='none', color=color,
                   label=label)
    for efm, (col, lab) in D.efm_dict.items():
        if efm in data.index:
            ax.plot(data.at[efm, x], gr[efm], markersize=5,
                    marker='o', color=col, label=None)
            ax.annotate(lab, xy=(data.at[efm, x], gr[efm]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', color=col)
    leg = ax.legend(loc='lower right', frameon=True)
    leg.get_frame().set_facecolor('#EEEEEE')

    for i, (d, ax) in enumerate(zip(plot_parameters, axS14.flat)):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)
        ax.set_xlim(-1e-3, None)
        ax.set_ylim(-1e-3, None)
        ax.set_ylabel(D.GROWTH_RATE_L)
        ax.set_xlabel(D.YIELD_L)

    figS14.tight_layout()
    D.savefig(figS14, 'S14')

    # %% SI Figure 15 - create protein allocation pie charts of selected EFMs

    # focus only on the selected EFMs, and rename the columns according
    # to the 3-letter acronyms
    efms = D.efm_dict.keys()
    _, efm_names = zip(*map(D.efm_dict.get, efms))

    # load data for the pie charts from the pareto plot
    rates_df, params_df, km_df, enzyme_abundance_df = \
        get_concatenated_raw_data('standard')

    # calculate the total cost of metabolic enzymes
    # it is given in hours, for the time required from the biomass reaction
    # to produce a mass equal to the mass of metabolic enzymes
    E_i = enzyme_abundance_df.loc[efms, :].mul(params_df['weight'].fillna(0))
    E_i.rename(index=dict(zip(efms, efm_names)), inplace=True)
    E_met = E_i.sum(1)  # total metabolic enzyme in grams per EFM
    v_BM = D.BIOMASS_MW * rates_df.loc[efms, D.R_BIOMASS] * D.SECONDS_IN_HOUR
    v_BM.rename(index=dict(zip(efms, efm_names)), inplace=True)

    # the growth rate in [1/h] if the biomass was 100% metabolic enzymes
    r_BM = v_BM / E_met

    n_fig_rows = int(np.ceil((len(D.efm_dict))/2.0))

    figS15, axS15 = plt.subplots(n_fig_rows, 2, figsize=(10, 5 * n_fig_rows))
    for ax, efm in zip(axS15.flat, efm_names):
        E_i_efm = E_i.loc[efm, :].sort_values(ascending=False)
        E_i_efm = E_i_efm / E_i_efm.sum()

        E_lumped = E_i_efm.drop(E_i_efm[E_i_efm.cumsum() > 0.95].index)
        E_lumped.loc[D.REMAINDER_L] = E_i_efm[E_i_efm.cumsum() > 0.95].sum()

        E_lumped.name = ''
        E_lumped.plot.pie(colors=list(map(D.reaction_to_rgb, E_lumped.index)),
                          labels=list(map(D.GET_REACTION_NAME, E_lumped.index)),
                          ax=ax)
        ax.set_title(r'\textbf{%s}' % efm + '\n' +
                     D.TOT_ENZYME_L + ' = %.2f' % (1.0/r_BM[efm]))

    D.savefig(figS15, 'S15')

    # %% SI Figure 16 - allocation area plots for glucose sweep
    rates_df, full_df = get_concatenated_raw_data('sweep_glucose')

    efms = D.efm_dict.keys()

    figS16, axS16 = plt.subplots(len(efms), 4, figsize=(20, 4 * len(efms)))
    for i, efm in enumerate(efms):
        df = full_df[full_df['efm'] == efm]
        if df.shape[0] == 0:
            continue
        v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * rates_df.at[efm, D.R_BIOMASS]

        # make a new DataFrame where the index is the glucose concentration
        # and the columns are the reactions and values are the costs.
        absol = full_df[full_df['efm'] == efm].pivot(index=full_df.columns[1],
                                                     columns='reaction',
                                                     values='E_i')
        D.allocation_area_plot(absol/v_BM, axS16[i, 0], axS16[i, 1],
                               xlabel='external glucose level [mM]')
        axS16[i, 0].annotate(D.efm_dict[efm][1], xy=(0.04, 0.95),
                             xycoords='axes fraction', ha='left', va='top',
                             size=20)

    # allocation area plots for oxygen sweep
    rates_df, full_df = get_concatenated_raw_data('sweep_oxygen')

    efms = D.efm_dict.keys()
    reactions = list(rates_df.columns)

    for i, efm in enumerate(efms):
        df = full_df[full_df['efm'] == efm]
        if df.shape[0] == 0:
            continue

        v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * rates_df.at[efm, D.R_BIOMASS]

        # make a new DataFrame where the index is the glucose concentration
        # and the columns are the reactions and values are the costs.
        absol = full_df[full_df['efm'] == efm].pivot(index=full_df.columns[1],
                                                     columns='reaction',
                                                     values='E_i')
        D.allocation_area_plot(absol/v_BM, axS16[i, 2], axS16[i, 3],
                               xlabel='O$_2$ level [mM]')
        axS16[i, 2].annotate(D.efm_dict[efm][1], xy=(0.04, 0.95),
                             xycoords='axes fraction', ha='left', va='top',
                             size=20)

    axS16[0, 1].set_title('Varying glucose levels', fontsize=25,
                          ha='right', va='bottom')
    axS16[0, 2].set_title('Varying oxygen levels', fontsize=25,
                          ha='left', va='bottom')
    figS16.tight_layout(h_pad=2.0)
    D.savefig(figS16, 'S16')

    # %% SI Figure 17 - Monod figure
    figS17 = plot_monod_figure(figure_data)
    D.savefig(figS17, 'S17')

    # %% SI Figure 18 - sensitivity to kcat of tpi (R6r)
    figS18, axS18 = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for i, ax in enumerate(axS18):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.98), xycoords='axes fraction',
                    fontsize=20, ha='left', va='top')

    axS18[0].set_title(r'effect of the $k_{cat}$ of \emph{tpi}')
    D.plot_dual_pareto(figure_data['standard'],
                       'std. $k_{cat}$ (7800 [$s^{-1}$])',
                       figure_data['low_kcat_r6r'],
                       'low $k_{cat}$ (7.8 [$s^{-1}$])',
                       s=8,
                       ax=axS18[0], x=D.YIELD_L, y=D.GROWTH_RATE_L,
                       draw_lines=False)
    axS18[0].set_xlim(0, None)
    axS18[0].legend(loc='upper center', fontsize=12)

    s = Sensitivity('standard')
    s.plot_sensitivity_as_errorbar(axS18[1], 'R6r', foldchange=2)
    axS18[1].set_xlim(0, None)
    axS18[1].set_title(r'sensitivity to 2-fold change in $k_{cat}$')

    maxy = figure_data['sweep_kcat_r6r'].max().max() * 1.2
    D.plot_sweep(figure_data['sweep_kcat_r6r'], r'$k_{cat}$ [$s^{-1}$]',
                 efm_dict=D.efm_dict, ax=axS18[2], legend_loc='center left',
                 legend_fontsize=10)
    axS18[2].set_xscale('log')
    axS18[2].set_ylim(0, maxy)
    axS18[2].fill_between([7837/2.0, 7837*2.0], 0, maxy,
                          color=(0.9, 0.9, 0.9))
    axS18[2].plot([7837, 7837], [0.0, maxy], '--',
                  color='grey', linewidth=1)
    axS18[2].text(7837, maxy*1.01, r'std. $k_{cat}$', ha='center',
                  color='grey')
    axS18[2].plot([7.837, 7.837], [0.0, maxy], '--',
                  color='grey', linewidth=1)
    axS18[2].text(7.837, maxy*1.01, r'low $k_{cat}$', ha='center',
                  color='grey')

    figS18.tight_layout()
    D.savefig(figS18, 'S18')

    # %%
    figS19 = plt.figure(figsize=(10, 10))
    axS19a = figS19.add_subplot(2, 2, 1, projection='3d')
    axS19b = figS19.add_subplot(2, 2, 2, projection='3d')
    axS19c = figS19.add_subplot(2, 2, 3, projection='3d')
    axS19d = figS19.add_subplot(2, 2, 4, projection='3d')
    plot_surface(axS19a, figure_data['standard'], c=D.GROWTH_RATE_L,
                 cmap='Oranges', vmax=0.7,
                 sweep_cache_fname='sweep2d_win_200x200.csv')
    plot_surface_diff(axS19b, ko_cache_fname='sweep2d_edko_win_200x200.csv')
    plot_surface_diff(axS19c, ko_cache_fname='sweep2d_empko_win_200x200.csv')
    plot_surface_diff(axS19d, ko_cache_fname='sweep2d_oxphosko_win_200x200.csv')
    axS19a.set_title('wild-type')
    axS19b.set_title('ED knockout')
    axS19c.set_title('EMP knockout')
    axS19d.set_title('OxPhos knockout')
    axS19a.set_zlim(0, 1)
    axS19b.set_zlim(0, 1)
    axS19c.set_zlim(0, 1)
    axS19d.set_zlim(0, 1)
    figS19.tight_layout(h_pad=2)
    D.savefig(figS19, 'S19')

    # %% S20 and S21 - epistasis plots
    e = Epistasis(figure_data)
    figS20 = e.plot_gr_epistasis()
    figS20.savefig(os.path.join(D.OUTPUT_DIR, 'FigS20.pdf'))

    figS21 = e.plot_yield_epistasis()
    D.savefig(figS21, 'S21')
    
    # %% convert EFM visualization flux plots from SVG to EPS
    with zipfile.ZipFile(D.ZIP_SVG_FNAME, 'r') as z:
        for efm, (_, efm_name) in D.efm_dict.items():
            with tempfile.NamedTemporaryFile(delete=True, suffix='.svg') as tmp_fp:
                # extract the SVG file from the ZIP file and save it as a temp
                # file.
                with z.open('efm%04d.svg' % efm, 'r') as svg_fp:
                    shutil.copyfileobj(svg_fp, tmp_fp)
                
                # then use inkscape to convert the SVG to EPS
                eps_fname = os.path.join(D.OUTPUT_DIR, 'FigS22_%s.eps' % efm_name)
                os.system('inkscape %s -E %s' % (tmp_fp.name, eps_fname))

    # %% fig S25 - glucose sweep at anaerobic conditions
    # find the "winning" EFM for each glucose level and make a color-coded
    # plot like the 3D surface plots
    
    def plot_1D_sweep(interpolated_df, ax0, ax1, ax2, color_func=None):
        best_df = pd.DataFrame(index=interpolated_df.index,
                               columns=[D.GROWTH_RATE_L, 'best_efm', 'hexcolor'])
        best_df[D.GROWTH_RATE_L] = interpolated_df.max(axis=1)
        best_df['best_efm'] = interpolated_df.idxmax(axis=1)
        
        best_efms = sorted(best_df['best_efm'].unique())
        
        if color_func is None:
            color_dict = dict(zip(best_efms, D.cycle_colors(len(best_efms),
                                                            h0=0.02, s=1)))
            color_func = color_dict.get
            
        best_df['hexcolor'] = best_df['best_efm'].apply(color_func)
        ax0.plot(interpolated_df.index, interpolated_df, '-',
                 linewidth=1, alpha=0.2, color=(0.5, 0.5, 0.8))
        ax0.set_xscale('log')
        ax0.set_xlim(0.6e-4, 1.5e4)
        ax0.set_ylim(1e-3, 0.86)
        ax0.set_xlabel(D.GLU_COL)
        ax0.set_ylabel(D.GROWTH_RATE_L)
        
        d = list(zip(best_df.index, best_df[D.GROWTH_RATE_L]))
        segments = zip(d[:-1], d[1:])
        colors = list(best_df['hexcolor'].iloc[1:].values)
        
        for efm in best_efms:
            if efm in D.efm_dict:
                label = D.efm_dict[efm]['label']
            else:
                label = 'EFM %04d' % efm
            ax1.plot(interpolated_df.index, interpolated_df[efm],
                     label='_nolegend_', linestyle='-',
                     color=D.efm_to_hex(efm), linewidth=0.5, alpha=0.5)
            ax1.plot([0, 1], [-1, -1],
                     label=label,
                     color=color_func(efm), linewidth=2)
        coll = LineCollection(segments, colors=colors, linewidths=2)
        ax1.add_collection(coll)
        ax1.legend(loc='best', fontsize=12)
        
        ax1.set_xscale('log')
        ax1.set_xlim(0.6e-4, 1.5e4)
        ax1.set_ylim(1e-3, 0.86)
        ax1.set_xlabel(D.GLU_COL)
        ax1.set_ylabel(D.GROWTH_RATE_L)
        
        ax2.plot(best_df[D.GROWTH_RATE_L], best_df.index,
                 '-', color=(0.5, 0.5, 0.8), linewidth=2)
        ax2.set_xlim(0, 0.7)
        ax2.set_ylim(0, 0.1)
        ax2.set_xlabel(D.GROWTH_RATE_L)
        ax2.set_ylabel('residual ' + D.GLU_COL)

    low_o2 = 0.0115
    
    figS25, axs25 = plt.subplots(3, 3, figsize=(10, 10))
    axs25[0, 1].set_title('anaerobic')
    axs25[1, 1].set_title(r'low $O_2$ (%.1f $\mu$M)' % (low_o2*1e3))
    axs25[2, 1].set_title(r'std. $O_2$ (%g mM)' % D.STD_CONC['oxygen'])

    interpolated_df = get_anaerobic_glucose_sweep_df(figure_data)
    plot_1D_sweep(interpolated_df, axs25[0, 0], axs25[0, 1], axs25[0, 2], None)
    
    interpolated_df = get_glucose_sweep_df(oxygen_conc=low_o2)
    plot_1D_sweep(interpolated_df, axs25[1, 0], axs25[1, 1], axs25[1, 2], D.efm_to_hex)
    
    interpolated_df = get_glucose_sweep_df(oxygen_conc=D.STD_CONC['oxygen'])
    plot_1D_sweep(interpolated_df, axs25[2, 0], axs25[2, 1], axs25[2, 2], D.efm_to_hex)
    
    for i, ax in enumerate(axs25.flat):
        ax.annotate(chr(ord('a')+i), xy=(0.02, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)
        
    figS25.tight_layout()
    D.savefig(figS25, 'S25')


    # %% measured protein abundances (if available)
    # S26 - correlation between EFM protein abundance predictions and
    rates_df, params_df, km_df, enzyme_abundance_df = \
        get_concatenated_raw_data('standard')

    # calculate correlation coefficients between the enzyme abundances and
    # the measured abundances (from Schmidt et al. 2015, glucose batch)
    X = enzyme_abundance_df.transpose()

    # in order to convert the enzyme abundances to realistic values, we need
    # to scale by a factor of 0.004 (see SI text, section S2.5)
    X *= 0.004

    y = map(D.PROTEOME_DICT.get, enzyme_abundance_df.columns)
    X['measured'] = pd.Series(index=enzyme_abundance_df.columns, data=y)
    X_pred = X.iloc[:, 0:-1].as_matrix()
    X_meas = X.iloc[:, -1].as_matrix()

    data = figure_data['standard'].copy()
    CORR_ENZ_L = 'enzyme abundance correlation'
    data[CORR_ENZ_L] = X.corr('spearman').loc['measured']

    # replace all zeros with a minimum protein level of 1 nM,
    # which represents the noise level (~1 molecule per cell)
    RMSE_ENZ_L = 'enzyme abundance RMSE'
    y = np.tile(X_meas, (X_pred.shape[1], 1))
    data[RMSE_ENZ_L] = np.sqrt(np.square(X_pred.T - y).mean(1))

    figS26 = plt.figure(figsize=(12, 5))
    axS26a = figS26.add_subplot(1, 2, 1)
    axS26a.set_title('Spearman correlation')
    axS26b = figS26.add_subplot(1, 2, 2)
    axS26b.set_title('exp')

    D.plot_basic_pareto(data, axS26a, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                        c=CORR_ENZ_L, cmap='copper_r')
    for efm in D.efm_dict.keys():
        xy = np.array(data.loc[efm, [D.YIELD_L, D.GROWTH_RATE_L]].tolist())
        xytext = xy + np.array((0, 0.07))
        axS26a.annotate(xy=xy, s=D.efm_dict[efm]['label'],
                        xycoords='data', xytext=xytext, ha='center',
                        arrowprops=dict(facecolor='black',
                        shrink=0.05, width=2, headwidth=4))
    axS26a.set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
    axS26a.set_ylim(-1e-3, 1.15*data[D.GROWTH_RATE_L].max())

    X[X == 0] = 1e-5
    X.fillna(1e-5, inplace=True)
    axS26b.plot(X.loc[:, 'measured'], X.loc[:, 9999], 'o', alpha=0.3)
    axS26b.plot([1e-5, 1], [1e-5, 1], 'b--')
    for i in X.index:
        xy = np.array(X.loc[i, ['measured', 9999]].tolist())
        axS26b.text(xy[0], xy[1], i, fontsize=8, ha='center', va='bottom')
    axS26b.set_xscale('log')
    axS26b.set_yscale('log')
    axS26b.set_ylabel('predicted enzyme abundance [mM]')
    axS26b.set_xlabel('measured enzyme abundance [mM]')

    D.savefig(figS26, 'S26')

    # %% glucose sweeps for low and high oxygen levels
    
    figS27, axs27 = plt.subplots(1, 3, figsize=(12, 4))
    low_o2 = 0.009
    std_o2 = D.STD_CONC['oxygen']
    axs27[0].set_title('low $O_2$ (%g mM)' % low_o2)
    axs27[1].set_title('std. $O_2$ (%g mM)' % std_o2)

    plot_glucose_sweep(axs27[0], oxygen_conc=low_o2,
                       ylim=(0, 0.86), legend_loc='upper center',
                       mark_glucose=False)
    plot_glucose_sweep(axs27[1], oxygen_conc=std_o2,
                       ylim=(0, 0.86), legend_loc=None,
                       mark_glucose=False)
    
    glu_grid = np.logspace(-4, -1, 200)
    interp_data_df = pd.DataFrame(index=glu_grid, columns=D.efm_dict.keys())
    interpolator = SweepInterpolator.interpolate_2D_sweep(D.efm_dict.keys())
    for efm in [1565]:
        gr = [interpolator.calc_gr(efm, g, std_o2)
              for g in glu_grid]
        axs27[2].plot(gr, glu_grid, '-', color=D.efm_dict[efm]['color'])


    axs27[2].set_xlabel(D.GROWTH_RATE_L)
    axs27[2].set_ylabel('residual ' + D.GLU_COL)
    axs27[2].set_title(r'std. $O_2$, only \emph{max-gr}')

    for i, ax in enumerate(axs27):
        ax.annotate(chr(ord('a')+i), xy=(0.02, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)

    figS27.tight_layout()
    D.savefig(figS27, 'S27')

    # %% A Pareto figure focusing only on the Pareto-optimal EFMs and 
    # neighboring points (generated by random sampling)
    figS28, axs28 = plt.subplots(1, 1, figsize=(5, 5)) 
    
    sampled_data = pd.read_pickle(pareto_sampling.PICKLE_FNAME)
    xdata = sampled_data[D.YIELD_L]
    ydata = sampled_data[D.GROWTH_RATE_L]
    sampled_data.plot.scatter(x=D.YIELD_L, y=D.GROWTH_RATE_L,
                              marker='.', c=(.9, .9, .9), s=10,
                              edgecolors=None, alpha=0.5, ax=axs28, zorder=1)
    axs28.set_xlabel(D.YIELD_L)
    axs28.set_ylabel(D.GROWTH_RATE_L)

    pareto_df = D.get_pareto(sampled_data, D.YIELD_L, D.GROWTH_RATE_L)
    pareto_df.plot.scatter(x=D.YIELD_L, y=D.GROWTH_RATE_L, marker='s',
                           color='k', s=15,
                           ax=axs28, legend=None, zorder=2)

    efm_data = figure_data['standard']
    efm_data.plot.scatter(x=D.YIELD_L, y=D.GROWTH_RATE_L,
                          marker='.', c=(1, 0.6, 0.6), s=15,
                          edgecolors=None, alpha=1, ax=axs28, zorder=3)

    pareto_df = D.get_pareto(efm_data, D.YIELD_L, D.GROWTH_RATE_L)
    pareto_df.plot.scatter(x=D.YIELD_L, y=D.GROWTH_RATE_L,
                           marker='s', c='r', s=30,
                           edgecolors=None, alpha=1, ax=axs28, zorder=4)
    for efm, (col, lab) in D.efm_dict.items():
        if efm in pareto_df.index:
            axs28.annotate(lab, xy=(pareto_df.at[efm, D.YIELD_L], pareto_df.at[efm, D.GROWTH_RATE_L]),
                           xytext=(5, 6), textcoords='offset points',
                           ha='left', va='bottom', color='r')
    axs28.set_xlim(18.5, 23.3)
    axs28.set_ylim(0.3, 0.8)

    D.savefig(figS28, 'S28')

    # %% A box plot of the capacity utilization across EFMs (grouped by reactions)
    figS29, axS29 = plt.subplots(2, 1, figsize=(10, 10))
    fig_name_and_titles = [('standard',  'glucose = %g mM, O$_2$ = %g mM' % 
                            (D.STD_CONC['glucoseExt'], D.STD_CONC['oxygen']),
                            axS29[0], True),
                           ('anaerobic', 'glucose = %g mM, no O$_2$' % 
                            (D.STD_CONC['glucoseExt']),
                            axS29[1], False)]
    for i, (fig_name, fig_title, ax, show_exp) in enumerate(fig_name_and_titles):
        ax.set_title(fig_title)
        ax.annotate(chr(ord('a')+i), xy=(0.01, 0.98), xycoords='axes fraction',
                    fontsize=20, ha='left', va='top')
        
        # read the raw files again, now including all kinetic parameters
        # and individual enzyme abundances at the ECM optimum
        zip_fnames, regex = D.DATA_FILES[fig_name]
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
        
        cap_util_median = caputil_df.groupby('reaction').median()
        order = cap_util_median.sort_values(by='capacity utilization').index

        # TODO: add the cap. util. of the "exp" flux mode in a different color
        # and also the measured one for glucose media (get from Dan)
        ax_box = sns.boxplot(x='reaction', y='capacity utilization',
                             data=caputil_df,
                             order=order, ax=ax)
        ax_box.set_xticklabels(order, rotation=90)
        ax_box.set_xlabel('')
        boxes = ax_box.artists
        for i, r in enumerate(order):
            boxes[i].set_facecolor(D.reaction_to_rgb(r))
            if show_exp:
                if r in caputil_exp.index:
                    ax.plot(i, caputil_exp.at[r, 'capacity utilization'],
                            markerfacecolor=(1, 0, 0), marker='o')
                else:
                    ax.plot(i, -0.05, markerfacecolor=(1, 0, 0), marker='.')
    figS29.tight_layout()
    D.savefig(figS29, 'S29')
    
