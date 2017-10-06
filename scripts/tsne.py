# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:39:26 2015

@author: noore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import definitions as D
from prepare_data import get_concatenated_raw_data

def plot_tsne_figure(figure_data, figsize=(15, 13)):
    data = figure_data['standard']
    # each one of the pareto zipfiles contains the rates of all the EFMs
    # so we arbitrarily chose Fig3_pareto to get them.

    rates_df, _, _ = get_concatenated_raw_data('standard')
    X = rates_df.as_matrix()

    model = TSNE(n_components=2)
    np.set_printoptions(suppress=True)
    X_new = model.fit_transform(X)

    rates_df_new = pd.DataFrame(index=rates_df.index, columns=('t-SNE dim 1', 't-SNE dim 2'))
    rates_df_new.iloc[:, 0] = X_new[:, 0]
    rates_df_new.iloc[:, 1] = X_new[:, 1]
    data = rates_df_new.join(data)

    #%%
    fig, axs = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
    axs = list(axs.flat)
    for i, ax in enumerate(axs):
        ax.annotate(chr(ord('a')+i), xy=(0.04, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20)

    xdata = rates_df_new.iloc[:, 0]
    ydata = rates_df_new.iloc[:, 1]
    axs[0].scatter(xdata, ydata, s=15, c=(0.2, 0.2, 0.7),
                   alpha=0.3)
    for efm in D.efm_dict.keys():
        xy = (xdata[efm], ydata[efm])
        axs[0].annotate(s=D.efm_dict[efm]['label'],
                        xy=xy, xycoords='data',
                        xytext=(30, 5), textcoords='offset points',
                        arrowprops=dict(facecolor='black',
                        shrink=0.05, width=2, headwidth=4),
                        ha='left', va='bottom')
    plot_parameters   = [
                         {'c': D.YIELD_L,       'title': 'biomass yield'},
                         {'c': D.GROWTH_RATE_L, 'title': 'growth rate'},
                         {'c': D.OXYGEN_L,      'title': 'oxygen uptake'},
                         {'c': D.ACE_L,         'title': 'acetate secretion'},
                         {'c': D.NH3_L,         'title': 'ammonia uptake'},
                         {'c': D.SUCCINATE_L,   'title': 'succinate secretion'},
                         {'c': D.ED_L,          'title': 'ED pathway'},
                         {'c': D.PPP_L,         'title': 'pentose phosphate pathway',},
                         ]

    for i, d in enumerate(plot_parameters):
        d['ax'] = axs[i+1]
        D.plot_basic_pareto(data, x=rates_df_new.columns[0],
                            y=rates_df_new.columns[1], c=d['c'],
                            ax=d['ax'], cmap='copper_r', linewidth=0.2, s=10)
        d['ax'].set_title(d['title'])
    fig.tight_layout()
    return fig

if __name__ == '__main__':
    figure_data = D.get_figure_data()
    fig = plot_tsne_figure(figure_data)
    fig.savefig(os.path.join(D.OUTPUT_DIR, 'FigS6.pdf'))
