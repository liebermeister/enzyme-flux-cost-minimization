# -*- coding: utf-8 -*-
"""
Created on Wed Oct 1 2015

@author: noore
"""

import re, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import definitions as D
from prepare_data import get_concatenated_raw_data
from phase_surface_plots import interpolate_single_condition

def epistasis_formula1(Wab, Wa, Wb):
    if min(Wa, Wb) == 0:
        return 1.0
    else:
        return Wab / min(Wa, Wb)

def epistasis_formula2(Wab, Wa, Wb):
    return Wab - Wa * Wb

def epistasis_formula3(Wab, Wa, Wb):
    if min(Wa, Wb) == 0:
        return 0.0
    else:
        return np.log(Wab / (Wa*Wb))

def epistasis_formula4(W_xy, W_x, W_y):
    """
        Segrè D, Deluna A, Church GM, Kishony R. Modular epistasis
        in yeast metabolism. Nat Genet. 2005;37: 77–83. doi:10.1038/ng1489
    """
    if W_xy > W_x*W_y:
        W_xy_tilde = min(W_x, W_y)
    else:
        W_xy_tilde = 0.0

    return (W_xy - W_x * W_y) / abs(W_xy_tilde - W_x*W_y)

epistasis_formula = epistasis_formula4

f_epi = lambda df, r1, r2 : \
    epistasis_formula(df.loc[r1, r2], df.loc[r1, r1], df.loc[r2, r2])

class Epistasis(object):

    def __init__(self, figure_data):
        rates_df, _, _, _ = get_concatenated_raw_data('standard')
        self.active_df = (rates_df == 0) # a boolean DataFrame of the active reactions in each EFM
        self.reactions = sorted(self.active_df.columns,
            key=lambda s: (int(re.findall('[rR]+(\d+)', s)[0]), s))

        yield_df = figure_data['standard'][D.YIELD_L]
        self.xticklabels = list(map(D.GET_REACTION_NAME, self.reactions))
        self.yticklabels = list(reversed(self.xticklabels))
        self.yd_double = pd.DataFrame(index=self.reactions,
                                      columns=self.reactions, dtype=float)
        self.yd_epistasis = pd.DataFrame(index=self.reactions,
                                         columns=self.reactions, dtype=float)
        max_yield = yield_df.max()
        for r1 in self.reactions:
            for r2 in self.reactions:
                inds = self.active_df[self.active_df[r1] & self.active_df[r2]].index
                self.yd_double.loc[r1, r2] = yield_df[inds].max() / max_yield
        self.yd_double.fillna(0, inplace=True)

        for r1 in self.reactions:
            for r2 in self.reactions:
                self.yd_epistasis.loc[r1, r2] = f_epi(self.yd_double, r1, r2)

        self.mask1 = np.zeros_like(self.yd_double, dtype=np.bool)
        self.mask1[np.triu_indices_from(self.mask1, 1)] = True
        self.mask2 = np.zeros_like(self.yd_epistasis, dtype=np.bool)
        self.mask2[np.triu_indices_from(self.mask2, 0)] = True

    def plot_yield_epistasis(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(self.yd_double, mask=self.mask1, ax=ax[0], cmap=D.GR_HEATMAP_CMAP,
                    square=True, linewidths=.2, cbar_kws={'shrink': .5})
        sns.heatmap(self.yd_epistasis, mask=self.mask2, ax=ax[1], cmap=D.EPISTATIS_CMAP,
                    square=True, linewidths=.2, cbar_kws={'shrink': .5})
        ax[0].annotate('a', xy=(0.75, 0.75), xycoords='axes fraction', fontsize=20,
                       ha='center', va='center')
        ax[0].set_title('Relative yield')
        ax[0].set_xticklabels(self.xticklabels, rotation=90, fontsize=8)
        ax[0].set_yticklabels(self.yticklabels, rotation=0, fontsize=8)
        ax[1].annotate('b', xy=(0.75, 0.75), xycoords='axes fraction', fontsize=20,
                       ha='center', va='center')
        ax[1].set_title('Scaled yield epistasis')
        ax[1].set_xticklabels(self.xticklabels, rotation=90, fontsize=8)
        ax[1].set_yticklabels(self.yticklabels, rotation=0, fontsize=8)
        fig.tight_layout()
        #self.yd_double.to_csv(os.path.join(D.OUTPUT_DIR, 'yield_double_ko.csv'))
        return fig

    def plot_gr_epistasis(self):
        figure_list = [['standard glucose (%g mM) and oxygen (%g mM)' %
                        (D.STD_CONC['glucoseExt'], D.STD_CONC['oxygen']),
                        D.STD_CONC['glucoseExt'], D.STD_CONC['oxygen']],
                       [r'low oxygen (%g mM)' % D.LOW_CONC['oxygen'],
                        D.STD_CONC['glucoseExt'], D.LOW_CONC['oxygen']],
                       [r'low glucose (%g mM)' % D.LOW_CONC['glucoseExt'],
                        D.LOW_CONC['glucoseExt'], D.STD_CONC['oxygen']]]

        fig, ax = plt.subplots(3, 2, figsize=(14, 18))
        for i, (fig_title, glucose, oxygen) in enumerate(figure_list):
            gr_df = interpolate_single_condition(glucose, oxygen)

            max_growth_rate = gr_df.max()

            gr_double = pd.DataFrame(index=self.reactions,
                                     columns=self.reactions, dtype=float)
            gr_epistasis = pd.DataFrame(index=self.reactions,
                                        columns=self.reactions, dtype=float)

            for r1 in self.reactions:
                for r2 in self.reactions:
                    inds = self.active_df[self.active_df[r1] & self.active_df[r2]].index
                    gr_double.loc[r1, r2] = gr_df[inds].max() / max_growth_rate
            gr_double.fillna(0, inplace=True)

            for r1 in self.reactions:
                for r2 in self.reactions:
                    gr_epistasis.loc[r1, r2] = f_epi(gr_double, r1, r2)

            sns.heatmap(gr_double, mask=self.mask1, ax=ax[i, 0],
                        cmap=D.GR_HEATMAP_CMAP,
                        square=True, linewidths=.2, cbar_kws={'shrink': .5})
            sns.heatmap(gr_epistasis, mask=self.mask2, ax=ax[i, 1], cmap=D.EPISTATIS_CMAP,
                        square=True, linewidths=.2, cbar_kws={'shrink': .5})
            ax[i, 0].annotate(chr(ord('a') + 2*i), xy=(0.75, 0.75),
                              xycoords='axes fraction', fontsize=20,
                              ha='center', va='center')
            ax[i, 0].set_title('Relative growth rates on\n' + fig_title)
            ax[i, 0].set_xticklabels(self.xticklabels, rotation=90, fontsize=8)
            ax[i, 0].set_yticklabels(self.yticklabels, rotation=0, fontsize=8)
            ax[i, 1].annotate(chr(ord('a') + 2*i + 1), xy=(0.75, 0.75),
                              xycoords='axes fraction', fontsize=20,
                              ha='center', va='center')
            ax[i, 1].set_title('Scaled growth rate epistasis on\n' + fig_title)
            ax[i, 1].set_xticklabels(self.xticklabels, rotation=90, fontsize=8)
            ax[i, 1].set_yticklabels(self.yticklabels, rotation=0, fontsize=8)
            #gr_double.to_csv(os.path.join(D.OUTPUT_DIR, 'gr_double_ko_%s.csv' % figname))

        fig.tight_layout()
        return fig

if __name__ == '__main__':
    figure_data = D.get_figure_data()
    e = Epistasis(figure_data)
    figS20 = e.plot_gr_epistasis()
    figS20.savefig(os.path.join(D.OUTPUT_DIR, 'FigS20.pdf'))

    figS21 = e.plot_yield_epistasis()
    figS21.savefig(os.path.join(D.OUTPUT_DIR, 'FigS21.pdf'))
