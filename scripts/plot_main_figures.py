# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:38:03 2015

@author: noore

Estimate what the maximal theoretical growth rate per EFM would be if all
enzymes were saturated (i.e. the kcat was realized).
"""

import os
import matplotlib.pyplot as plt
import definitions as D
from phase_surface_plots import allocation_pie_chart, plot_surface, \
                                plot_heatmap_diff, plot_heatmap, \
                                plot_oxygen_sweep, plot_oxygen_dual_pareto
from prepare_data import get_concatenated_raw_data
from mpl_toolkits.mplot3d import Axes3D # NOTE!!! keep this for the 3D plots

figure_data = D.get_figure_data()

if False:
    # %% Figure 2c
    fig2c, ax2c = plt.subplots(1, 1, figsize=(4.5, 4.5))

    data = figure_data['standard']
    # remove oxygen-sensitive EFMs
    data.loc[data[D.STRICTLY_ANAEROBIC_L], D.GROWTH_RATE_L] = 0
    D.plot_basic_pareto(data, ax2c, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                        efm_dict=D.efm_dict,
                        facecolors=D.PARETO_NEUTRAL_COLOR, edgecolors='none',
                        mark_pareto=False)
    ax2c.set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
    ax2c.set_ylim(-1e-3, 1.15*data[D.GROWTH_RATE_L].max())
    ax2c.set_title('glucose = %g mM, O$_2$ = %g mM' %
                   (D.STD_CONC['glucoseExt'], D.STD_CONC['oxygen']))
    fig2c.tight_layout()

    fig2c.savefig(os.path.join(D.OUTPUT_DIR, 'Fig2c.svg'))

    # %% Table 1
    # write the EFM data that is presented in table 1 (for the selected EFMs)
    data = figure_data['standard'].loc[D.efm_dict.keys(), :]
    data.insert(0, 'acronym', map(lambda x: x['label'], D.efm_dict.values()))
    data.to_csv(os.path.join(D.OUTPUT_DIR, 'Table1.csv'))

    # %% Figure 3
    fig3, ax3 = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    plot_parameters = [
        {'c': D.OXYGEN_L,     'short_title': 'O$_2$ uptake'},
        {'c': D.ACE_L,        'short_title': 'acetate secretion'},
        {'c': D.LACTATE_L,    'short_title': 'lactate secretion'},
        {'c': D.SUCCINATE_L,  'short_title': 'succinate secretion'},
    ]
    ax3 = list(ax3.flat)
    data = figure_data['standard']

    for i, d in enumerate(plot_parameters):
        d['ax'] = ax3[i]
        d['ax'].annotate(chr(ord('a')+i), xy=(0.02, 0.98),
                         xycoords='axes fraction', ha='left', va='top',
                         size=20)
        d['ax'].set_title(d['short_title'])
        d['ax'].set_xlim(-1e-3, 1.05*data[D.YIELD_L].max())
        d['ax'].set_ylim(-1e-3, 1.05*data[D.GROWTH_RATE_L].max())

        D.plot_basic_pareto(data, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                            c=d['c'], ax=d['ax'], cmap='copper_r')

    fig3.tight_layout(h_pad=0.2)
    fig3.savefig(os.path.join(D.OUTPUT_DIR, 'Fig3.pdf'))

    # %% Figure 4 - glucose & oxygen sweeps

    fig4 = plt.figure(figsize=(11, 15))

    ax4a = fig4.add_subplot(3, 2, 1, xscale='linear', yscale='linear')
    ax4b = fig4.add_subplot(3, 2, 2, xscale='log', yscale='linear', sharey=ax4a)
    ax4c = fig4.add_subplot(3, 2, 3, projection='3d')
    ax4d = fig4.add_subplot(3, 2, 4, projection='3d')
    ax4e = fig4.add_subplot(3, 2, 5, projection='3d')
    ax4f = fig4.add_subplot(3, 2, 6, projection='3d')

    for i, ax in enumerate([ax4a, ax4b, ax4c, ax4d, ax4e, ax4f]):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20, color='k')

    plot_oxygen_dual_pareto(figure_data['standard'], ax4a,
                            draw_lines=False, s=20)
    ax4b.set_ylim(0, 0.9)
    plot_oxygen_sweep(ax4b)

    plot_surface(ax4c, figure_data['standard'], c=D.GROWTH_RATE_L,
                 cmap='Oranges', vmax=0.7)
    plot_surface(ax4d, figure_data['standard'], c=D.OXYGEN_L, vmax=0.7)
    plot_surface(ax4e, figure_data['standard'], c=D.ACE_L, vmax=1.5)
    plot_surface(ax4f, figure_data['standard'], c=D.LACTATE_L, vmax=1.5)

    fig4.tight_layout(h_pad=3)
    fig4.savefig(os.path.join(D.OUTPUT_DIR, 'Fig4.pdf'))

    # %% histogram of all different EFM growth rates in a specific condition
    fig5 = plt.figure(figsize=(9, 4.8))
    ax5a = fig5.add_subplot(1, 2, 1)
    ax5b = fig5.add_subplot(1, 2, 2)

    efm = allocation_pie_chart(ax5a, D.STD_CONC['glucoseExt'],
                               D.STD_CONC['oxygen'])
    rates_df, full_df = get_concatenated_raw_data('sweep_glucose')

    df = full_df[full_df['efm'] == efm]
    v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * rates_df.at[efm, D.R_BIOMASS]

    # make a new DataFrame where the index is the glucose concentration
    # and the columns are the reactions and values are the costs.
    absol = full_df[full_df['efm'] == efm].pivot(index=full_df.columns[1],
                                                 columns='reaction',
                                                 values='E_i')
    D.allocation_area_plot(absol/v_BM, None, ax5b,
                           xlabel='external glucose level [mM]', n_best=14)

    ax5b.plot([D.STD_CONC['glucoseExt'], D.STD_CONC['glucoseExt']],
              [0.0, 1.0], '--', color='white', linewidth=1)
    ax5b.legend_.remove()

    ax5a.annotate('a', xy=(0.02, 0.95),
                  xycoords='axes fraction', ha='left', va='top',
                  size=20)
    ax5b.annotate('b', xy=(0.02, 0.95),
                  xycoords='axes fraction', ha='left', va='top',
                  size=20, color='white')

    fig5.tight_layout()
    fig5.savefig(os.path.join(D.OUTPUT_DIR, 'Fig5.pdf'))

    # %% histogram of all different EFM growth rates in a specific condition
    fig6, axs6 = plt.subplots(2, 2, figsize=(7.5, 6), sharey=True, sharex=True)
    fig6.subplots_adjust(wspace=0.4)

    plot_heatmap(axs6[0, 0])
    axs6[0, 0].set_title(r'wild-type')

    plot_heatmap_diff(axs6[0, 1], 'sweep2d_win_200x200.csv',
                      'sweep2d_edko_win_200x200.csv', vmax=0.5)
    axs6[0, 1].set_title(r'$\mu_{WT} / \mu_{\Delta ED}$')

    plot_heatmap_diff(axs6[1, 0], 'sweep2d_win_200x200.csv',
                      'sweep2d_empko_win_200x200.csv', vmax=0.5)
    axs6[1, 0].set_title(r'$\mu_{WT} / \mu_{\Delta EMP}$')

    plot_heatmap_diff(axs6[1, 1], 'sweep2d_edko_win_200x200.csv',
                      'sweep2d_empko_win_200x200.csv', vmax=0.5)
    axs6[1, 1].set_title(r'$\mu_{\Delta ED} / \mu_{\Delta EMP}$')

    arrowprops = dict(facecolor='black', width=1, headwidth=5, shrink=0.01)
    axs6[1, 1].annotate('$ED > EMP$', xy=(1.08, 0), xytext=(1.20, -0.1),
                        xycoords='axes fraction', ha='left', va='center',
                        size=8, color='k',
                        arrowprops=arrowprops)
    axs6[1, 1].annotate('$EMP > ED$', xy=(1.08, 1), xytext=(1.20, 1.1),
                        xycoords='axes fraction', ha='left', va='center',
                        size=8, color='k',
                        arrowprops=arrowprops)

    for i, ax in enumerate(axs6.flat):
        ax.annotate(chr(ord('a')+i), xy=(0.02, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20, color='k')

    axs6[0, 1].get_yaxis().set_visible(False)
    axs6[1, 1].get_yaxis().set_visible(False)
    axs6[0, 0].get_xaxis().set_visible(False)
    axs6[0, 1].get_xaxis().set_visible(False)

    fig6.savefig(os.path.join(D.OUTPUT_DIR, 'Fig6.pdf'))
