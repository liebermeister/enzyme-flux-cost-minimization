#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:51:06 2017

@author: eladn
"""

from prepare_data import read_pareto_zipfile
import os
import definitions as D
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    PICKLE_FNAME = os.path.join(D.TEMP_DIR, 'pareto_sampling.pkl')
    if not os.path.exists(PICKLE_FNAME):
        data_list = []
        for set_fname in ['n39-p80', 'n39-p81']:
            zip_fname = os.path.join(D.DATA_DIR, set_fname + '.zip')
            data_list.append(read_pareto_zipfile(zip_fname))
        data = pd.concat(data_list, axis=0, join='inner')
        data.to_pickle(PICKLE_FNAME)
    else:
        data = pd.read_pickle(PICKLE_FNAME)

    # the indexes of these DataFrames are not EFMs as usually happens in
    # other datasets, but just arbitraty sequence numbers. To avoid 
    # duplicates, we override them with new unique indexes
    data.index = range(data.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 
    
    xdata = data[D.YIELD_L]
    ydata = data[D.GROWTH_RATE_L]
    CS = ax.scatter(xdata, ydata, marker='o',
                    facecolors=(.8, .7, .7), edgecolors=None, alpha=0.2)
    ax.set_xlabel(D.YIELD_L)
    ax.set_ylabel(D.GROWTH_RATE_L)

    # find the EFMs which are on the pareto front and mark them
    pareto_idx = []
    for i in ydata.sort_values(ascending=False).index:
        if pareto_idx == [] or xdata[i] > xdata[pareto_idx[-1]]:
            pareto_idx.append(i)

    xpareto = xdata[pareto_idx]
    ypareto = ydata[pareto_idx]
    ax.plot(xpareto, ypareto, marker='.', linewidth=1, color=(0, 0, 0.5),
            markersize=20)
    
    fig.savefig(os.path.join(D.OUTPUT_DIR, 'Fig_pareto_sampling.pdf'))