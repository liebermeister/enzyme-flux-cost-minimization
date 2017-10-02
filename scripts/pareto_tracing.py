#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:08:13 2017

@author: eladn

A script for finding the Pareto-optimal EFMs and reporting
them in decreasing order of growth rate (and increasing yield)
"""

import os
import matplotlib.pyplot as plt
import definitions as D
from phase_surface_plots import allocation_pie_chart, plot_surface, \
                                plot_heatmap_diff, plot_heatmap, \
                                plot_oxygen_sweep, plot_oxygen_dual_pareto
from prepare_data import get_concatenated_raw_data

figure_data = D.get_figure_data()
data = figure_data['standard']

# find the EFMs which are Pareto optimal
pareto = []
for i in data.sort_values(D.GROWTH_RATE_L, ascending=False).index:
    if pareto == [] or data.at[i, D.YIELD_L] > data.at[pareto[-1], D.YIELD_L]:
        pareto.append(i)

pareto_data = data.loc[pareto, :]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
pareto_data.plot(kind='scatter', x=D.YIELD_L, y=D.GROWTH_RATE_L, marker='o')

export_df = data[[D.YIELD_L, D.GROWTH_RATE_L]]
export_df['pareto'] = False
export_df.loc[pareto, 'pareto'] = True
export_df.to_csv(os.path.join(D.OUTPUT_DIR, 'standard_pareto.csv'))