# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:29:21 2016

A script for drawing the Phase Plane plot for the
glucose/oxygen levels (for the 6 selected EFMs)

@author: noore
"""
import os
import pandas as pd
import numpy as np
from matplotlib import rcParams, cm, pyplot, colors, colorbar
import zipfile
import definitions as D
from prepare_data import read_sweep_zipfile, get_df_from_sweep_zipfile, \
                         get_general_parameters_from_zipfile
from scipy.interpolate import RectBivariateSpline
import itertools
import seaborn as sns

PREFIX = 'n39-p'
ITER_RANGE = range(31, 48)
REGEX = 'mext-glucoseExt-'

class SweepInterpolator(object):

    def __init__(self, data_df, efm_list):
        self.efm_list = efm_list or data_df['efm'].unique().tolist()

        # interpolate 2D function for each EFM
        self.f_interp_dict = {}
        max_growth_rate = 0
        for efm in self.efm_list:
            try:
                gr_df = data_df[data_df['efm'] == efm].pivot(index=D.GLU_COL,
                                                             columns=D.OX_COL,
                                                             values=D.GROWTH_RATE_L)

                f = RectBivariateSpline(np.log10(gr_df.index),
                                        np.log10(gr_df.columns),
                                        gr_df)
                self.f_interp_dict[efm] = f
                max_gr_efm = f(np.log10(D.MAX_CONC['glucoseExt']),
                               np.log10(D.MAX_CONC['oxygen']))[0, 0]
                max_growth_rate = max(max_growth_rate, max_gr_efm)
            except ValueError:
                print("WARNING: cannot interpolate 2D function for EFM #%04d" % efm)

    def calc_gr(self, efm, glucose, oxygen):
        return self.f_interp_dict[efm](np.log10(glucose), np.log10(oxygen))[0, 0]

    def calc_all_gr(self, glucose, oxygen):
        data = [self.calc_gr(efm, glucose, oxygen) for efm in self.efm_list]
        return pd.Series(index=self.efm_list, data=data)

    @staticmethod
    def get_general_params(iter_num):
        prefix = '%s%d' % (PREFIX, iter_num)

        zip_fname = os.path.join(D.DATA_DIR, '%s.zip' % prefix)

        # get the oxygen level from the "metfixed.csv" file inside the zipfile
        with zipfile.ZipFile(zip_fname, 'r') as z:
            rates_df, params_df, km_df = \
                get_general_parameters_from_zipfile(z, prefix)

        return rates_df, params_df, km_df

    @staticmethod
    def get_efm_list_for_knockout(ko):
        """
            get the descriptions of the EFMs from one of the sweep
            files (doesn't matter which)
        """
        rates_df, _, _ = SweepInterpolator.get_general_params(min(ITER_RANGE))
        if type(ko) != list:
            ko = [ko]
        efms_to_keep = rates_df[~rates_df[ko].any(1)].index
        return list(efms_to_keep)

    def calculate_growth_on_grid(self, ko=None, N=200):
        """
            use the interpolations to calculate the growth rate on a NxN grid
        """
        glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                               np.log10(D.MAX_CONC['glucoseExt']), N)
        ox_grid  = np.logspace(np.log10(D.MIN_CONC['oxygen']),
                               np.log10(D.MAX_CONC['oxygen']), N)

        if ko:
            efms = SweepInterpolator.get_efm_list_for_knockout(ko)
        else:
            efms = self.efm_list

        monod_df = []
        for g, o in itertools.product(glu_grid, ox_grid):
            growth_rates = [(self.calc_gr(efm, g, o), efm) for efm in efms]
            growth_rates.sort(reverse=True)
            best_gr, best_efm = growth_rates[0]
            monod_df.append((g, o, best_efm, best_gr))

        monod_df = pd.DataFrame(monod_df,
                                columns=[D.GLU_COL, D.OX_COL, 'best_efm', D.GROWTH_RATE_L])

        return monod_df

    @staticmethod
    def interpolate_2D_sweep(efm_list=None):
        """
            Combine all glucose and oxygen sweeps into one DataFrame
        """
        data_df = get_complete_sweep_data()
        return SweepInterpolator(data_df, efm_list)


def get_raw_sweep_data(iter_num):
    prefix = '%s%d' % (PREFIX, iter_num)

    zip_fname = os.path.join(D.DATA_DIR, '%s.zip' % prefix)

    # get the oxygen level from the "metfixed.csv" file inside the zipfile
    with zipfile.ZipFile(zip_fname, 'r') as z:
        ox_df = pd.read_csv(z.open('%s/metfixed.csv' % prefix, 'r'),
                                      header=None, index_col=0)
        ox_conc = ox_df.at['oxygen', 1] # in mM

    _, df = get_df_from_sweep_zipfile(zip_fname, REGEX)
    df.rename(columns={REGEX: D.GLU_COL}, inplace=True)
    df.insert(2, D.OX_COL, float(ox_conc))
    return df

def get_sweep_data(iter_num):
    prefix = '%s%d' % (PREFIX, iter_num)

    zip_fname = os.path.join(D.DATA_DIR, '%s.zip' % prefix)

    # get the oxygen level from the "metfixed.csv" file inside the zipfile
    with zipfile.ZipFile(zip_fname, 'r') as z:
        ox_df = pd.read_csv(z.open('%s/metfixed.csv' % prefix, 'r'),
                                      header=None, index_col=0)
        ox_conc = ox_df.at['oxygen', 1] # in mM

    df = read_sweep_zipfile(zip_fname, REGEX)
    df = pd.melt(df.reset_index(), id_vars='efm', value_name=D.GROWTH_RATE_L)
    df.rename(columns={REGEX: D.GLU_COL}, inplace=True)
    df.insert(2, D.OX_COL, float(ox_conc))
    return df

def cache_complete_sweep_data():
    df_list = []
    for iter_num in ITER_RANGE:
        df_list.append(get_sweep_data(iter_num))
    data_df = pd.concat(df_list)

    data_df.sort_values(['efm', D.GLU_COL, D.OX_COL], inplace=True)
    data_df = data_df[['efm', D.GLU_COL, D.OX_COL, D.GROWTH_RATE_L]]
    data_df[D.GLU_COL] = pd.to_numeric(data_df[D.GLU_COL])
    data_df[D.OX_COL] = pd.to_numeric(data_df[D.OX_COL])
    data_df.to_csv(os.path.join(D.TEMP_DIR, 'sweep2d_gr.csv'))

def get_complete_sweep_data():
    sweep_cache_fname = os.path.join(D.TEMP_DIR, 'sweep2d_gr.csv')
    if not os.path.exists(sweep_cache_fname):
        cache_complete_sweep_data()
    return pd.read_csv(sweep_cache_fname)

def get_winning_enzyme_allocations():
    df_list = []
    for iter_num in ITER_RANGE:
        df_std_ox = get_raw_sweep_data(iter_num)
        df_std_ox_gr = get_sweep_data(iter_num)

        # find the winning EFM in each condition (glucose level)
        winning = df_std_ox_gr.sort_values(D.GROWTH_RATE_L, ascending=False).groupby(D.GLU_COL).first().reset_index()

        # merge the winning table with the enzyme data table, so that only the
        # enzyme allocation data for the winning EFM in each condition is kept
        win_enz_df = pd.merge(df_std_ox, winning, on=['efm', D.GLU_COL, D.OX_COL], how='inner')
        df_list.append(win_enz_df)
    df = pd.concat(df_list)
    return df

def write_cache_files():
    """
        write all relevant cache files
    """

    # 1) the growth rates for each triplet: EFM, glucose, oxygen
    cache_complete_sweep_data()

    # 2) for each glucose and oxygen pair, find the EFM with the maximal
    #    growth rate, and keep only its enzyme allocation values
    sweep2d_win_enzymes = get_winning_enzyme_allocations()
    sweep2d_win_enzymes.to_csv(os.path.join(D.TEMP_DIR, 'sweep2d_win_enzymes.csv'))

    # 3) after interpolating the g.r. for each EFM over a 200x200 2D grid
    #    find the EFM with the maximal growth rate (best_efm)
    f_interp_dict = SweepInterpolator.interpolate_2D_sweep()

    kos = [(None, None),
           ('R60', 'ed'),
           ('R3', 'emp'),
           (D.R_OXYGEN_DEPENDENT, 'oxphos')]

    for ko, name in kos:
        if name is None:
            fname = os.path.join(D.TEMP_DIR, 'sweep2d_win_200x200.csv')
        else:
            fname = os.path.join(D.TEMP_DIR, 'sweep2d_%sko_win_200x200.csv' % name)
        sweep2d_grid = f_interp_dict.calculate_growth_on_grid(ko)
        sweep2d_grid.to_csv(fname)

def interpolate_single_condition(glucose=None, oxygen=None):
    interpolator = SweepInterpolator.interpolate_2D_sweep()
    glucose = glucose or D.STD_CONC['glucoseExt']
    oxygen = oxygen or D.STD_CONC['oxygen']
    data_df = interpolator.calc_all_gr(glucose, oxygen)
    return data_df

def plot_growth_rate_hist(glucose=None, oxygen=None, ax=None):
    glucose = glucose or D.STD_CONC['glucoseExt']
    oxygen = oxygen or D.STD_CONC['oxygen']
    data_df = interpolate_single_condition(glucose, oxygen)

    if ax is not None:
        bins = np.linspace(0, 0.8, 20)
        sns.distplot(data_df, ax=ax, bins=bins,
                     color=D.BAR_COLOR, kde=False)
        ax.set_title('[glu] = %g mM, [O$_2$] = %g mM' % (glucose, oxygen))
        ax.set_xlabel(D.GROWTH_RATE_L)
        ax.set_ylabel('no. of EFMs')
        ax.set_xlim(0, None)

def allocation_pie_chart(ax, glucose=100.0, oxygen=3.7e-3):
    win_enz_df = pd.read_csv(
        os.path.join(D.TEMP_DIR, 'sweep2d_win_enzymes.csv'))

    glu = sorted(win_enz_df[D.GLU_COL].unique(), key=lambda x: (x-glucose)**2)[0]
    ox = sorted(win_enz_df[D.OX_COL].unique(), key=lambda x: (x-oxygen)**2)[0]

    enz = win_enz_df[(win_enz_df[D.GLU_COL] == glu) & (win_enz_df[D.OX_COL] == ox)]
    efm = enz['efm'].unique()[0]
    gr = enz[D.GROWTH_RATE_L].unique()[0]
    E_i = enz.set_index('reaction')['E_i'].sort_values(ascending=False)
    E_i = E_i / E_i.sum()

    E_lumped = E_i.drop(E_i[E_i.cumsum() > 0.95].index)
    E_lumped.loc[D.REMAINDER_L] = E_i[E_i.cumsum() > 0.95].sum()

    E_lumped.name = ''
    E_lumped.plot.pie(colors=list(map(D.reaction_to_rgb, E_lumped.index)),
                      labels=list(map(D.GET_REACTION_NAME, E_lumped.index)),
                      ax=ax)

    if efm in D.efm_dict:
        efm_name = D.efm_dict[efm]['label']
    else:
        efm_name = '%d' % efm
    ax.set_title('[glu] = %g mM, [O$_2$] = %g mM\nbest EFM is %s, %s = %.2f' %
                 (glucose, oxygen, efm_name, D.GROWTH_RATE_L, gr))
    return efm


def plot_surface(ax, figdata,
                 z=D.GROWTH_RATE_L, c=D.GROWTH_RATE_L, cmap=None, vmax=None,
                 sweep_cache_fname='sweep2d_win_200x200.csv'):
    """
        plot a 3D surface plot of the 2D-sweep axes, with growth rate (by default)
        as the z-axis. One can either use color to indicate hight, or overlay the
        mesh with another color based on a 4th parameter.
    """
    monod_df, axis_params = get_monod_data(sweep_cache_fname)
    #X = np.log10(monod_df[D.GLU_COL].as_matrix().reshape(200, 200).T)
    #Y = np.log10(monod_df[D.OX_COL].as_matrix().reshape(200, 200).T)
    monod_df = monod_df.join(figdata, on='best_efm', rsuffix='_')

    X = np.arange(0, axis_params[D.GLU_COL]['N'])
    Y = np.arange(0, axis_params[D.OX_COL]['N'])
    X, Y = np.meshgrid(X, Y)

    # create matrix-style DataFrames for the growth rate and oxygen uptake rate
    z_mat = monod_df.pivot(index=D.GLU_COL, columns=D.OX_COL, values=z).T.as_matrix()
    cmap = cmap or cm.magma_r
    if z == c: # make a standard surface plot with gridlines and big strides
        vmax = vmax or z_mat.max().max()
        ax.plot_surface(X, Y, z_mat, rstride=1, cstride=1, cmap=cmap,
                        antialiased=False, rasterized=True,
                        linewidth=0, vmin=0, vmax=vmax, shade=False)
    else:      # use a different matrix for the color coding of the surface
        c_mat = monod_df.pivot(index=D.GLU_COL, columns=D.OX_COL, values=c).T.as_matrix()
        vmax = vmax or c_mat.max().max()
        c_colors = np.empty((X.shape[1], X.shape[0], 4), dtype=float)
        for ox in range(X.shape[1]):
            for gl in range(X.shape[0]):
                c_colors[ox, gl, :] = cmap(c_mat[ox, gl] / vmax)
        ax.plot_surface(X, Y, z_mat, facecolors=c_colors,
                        antialiased=False, rasterized=True,
                        rstride=1, cstride=1, linewidth=0, shade=False)

        sm = cm.ScalarMappable(cmap=cmap, norm=pyplot.Normalize(vmin=0, vmax=vmax))
        sm._A = []
        pyplot.colorbar(sm, ax=ax, fraction=0.07, shrink=0.5, label=c)

    ax.plot_wireframe(X, Y, z_mat, rstride=6, cstride=6, linewidth=0.2,
                      edgecolor='k', alpha=0.3)
    ax.set_xticks(axis_params[D.GLU_COL]['xticks'])
    ax.set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
    ax.set_yticks(axis_params[D.OX_COL]['xticks'])
    ax.set_yticklabels(axis_params[D.OX_COL]['xticklabels'])
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(D.OX_COL)
    ax.set_zlabel(z, rotation=90)
    ax.view_init(20, -120)


def plot_surface_diff(ax, ko_cache_fname, wt_cache_fname='sweep2d_win_200x200.csv'):

    monod_df, axis_params = get_monod_data(wt_cache_fname)
    wt_gr_mat = monod_df.pivot(index=D.GLU_COL, columns=D.OX_COL,
                               values=D.GROWTH_RATE_L).T.as_matrix()

    monod_df, axis_params = get_monod_data(ko_cache_fname)
    ko_gr_mat = monod_df.pivot(index=D.GLU_COL, columns=D.OX_COL,
                               values=D.GROWTH_RATE_L).T.as_matrix()

    X = np.arange(0, axis_params[D.GLU_COL]['N'])
    Y = np.arange(0, axis_params[D.OX_COL]['N'])
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, ko_gr_mat,
                    rstride=6, cstride=6, cmap='Oranges',
                    linewidth=0.25, edgecolors='r',
                    vmin=0, vmax=0.7)

    ax.plot_wireframe(X, Y, wt_gr_mat,
                      rstride=6, cstride=6, linewidth=0.5,
                      colors=(0.1, 0.1, 0.6), alpha=1)

    ax.set_xticks(axis_params[D.GLU_COL]['xticks'])
    ax.set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
    ax.set_yticks(axis_params[D.OX_COL]['xticks'])
    ax.set_yticklabels(axis_params[D.OX_COL]['xticklabels'])
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(D.OX_COL)
    ax.set_zlabel(D.GROWTH_RATE_L, rotation=90)
    ax.view_init(20, -120)


def plot_heatmap_diff(ax, cache_fname1, cache_fname2, vmax=1):
    monod1_df, wt_axis_params = get_monod_data(cache_fname1)
    gr1_mat = monod1_df.pivot(index=D.GLU_COL, columns=D.OX_COL, values=D.GROWTH_RATE_L).T

    monod2_df, ko_axis_params = get_monod_data(cache_fname2)
    gr2_mat = monod2_df.pivot(index=D.GLU_COL, columns=D.OX_COL, values=D.GROWTH_RATE_L).T

    pcol = ax.imshow(np.log2(gr1_mat) - np.log2(gr2_mat),
                     interpolation='none', cmap='bwr', vmin=-vmax, vmax=vmax,
                     origin='lower', aspect=1)
    pyplot.colorbar(pcol, ax=ax, label=r'log$_2$ fold change',
                    fraction=0.1)
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(D.OX_COL)
    ax.set_xticks(wt_axis_params[D.GLU_COL]['xticks'])
    ax.set_xticklabels(wt_axis_params[D.GLU_COL]['xticklabels'])
    ax.set_yticks(wt_axis_params[D.OX_COL]['xticks'])
    ax.set_yticklabels(wt_axis_params[D.OX_COL]['xticklabels'])

def plot_heatmap(ax, wt_cache_fname='sweep2d_win_200x200.csv', vmax=None):
    wt_monod_df, wt_axis_params = get_monod_data(wt_cache_fname)
    wt_gr_mat = wt_monod_df.pivot(index=D.GLU_COL,
                                  columns=D.OX_COL, values=D.GROWTH_RATE_L).T

    pcol = ax.imshow(wt_gr_mat,
                     interpolation='none', cmap='magma_r', vmin=0, vmax=vmax,
                     origin='lower', aspect=1)
    pyplot.colorbar(pcol, ax=ax, label=r'growth rate [h$^-1$]',
                    fraction=0.1)
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(D.OX_COL)
    ax.set_xticks(wt_axis_params[D.GLU_COL]['xticks'])
    ax.set_xticklabels(wt_axis_params[D.GLU_COL]['xticklabels'])
    ax.set_yticks(wt_axis_params[D.OX_COL]['xticks'])
    ax.set_yticklabels(wt_axis_params[D.OX_COL]['xticklabels'])

def get_monod_data(sweep_cache_fname='sweep2d_win_200x200.csv'):
    monod_df = pd.read_csv(os.path.join(D.TEMP_DIR, sweep_cache_fname))

    # construct the bitmap by assigning the color of the winning EFM to each
    # pixel
    monod_df['hexcolor'] = monod_df['best_efm'].apply(D.efm_to_hex)

    standard_conc = {D.GLU_COL: 100.0, D.OX_COL: 3.7}
    ticks = {D.GLU_COL: [0, 44, 88, 133, 177],
             D.OX_COL: [0, 50, 100, 150, 199]}
    axis_params = {}
    for col in [D.GLU_COL, D.OX_COL]:
        axis_params[col] = {}
        levels = sorted(monod_df[col].unique())
        axis_params[col]['N'] = len(levels)
        axis_params[col]['min'] = monod_df[col].min()
        axis_params[col]['max'] = monod_df[col].max()
        x_std = np.log(standard_conc[col])
        x_min = np.log(monod_df[col].min())
        x_max = np.log(monod_df[col].max())
        axis_params[col]['std_ind'] = len(levels) * (x_std - x_min) / (x_max - x_min)
        axis_params[col]['xticks'] = ticks[col]
        tickvalues = [levels[i] for i in ticks[col]]
        axis_params[col]['xticklabels'] = map(D.as_base10_exp, tickvalues)
    return monod_df, axis_params

def plot_monod_surface(figure_data, sweep_cache_fname='sweep2d_win_200x200.csv'):
    monod_df, axis_params = get_monod_data(sweep_cache_fname)
    max_growth_rate = monod_df[D.GROWTH_RATE_L].max()

    figS12, axS12 = pyplot.subplots(3, 3, figsize=(12, 12))
    cbar_ax = figS12.add_axes([.72, .75, .02, .2])

    # create a bitmap to be used with imshow
    hexcolor_df = monod_df.pivot(index=D.GLU_COL,
                                 columns=D.OX_COL,
                                 values='hexcolor')
    best_efm_color = np.zeros((axis_params[D.OX_COL]['N'],
                               axis_params[D.GLU_COL]['N'], 3))
    for i, g in enumerate(hexcolor_df.index):
        for j, o in enumerate(hexcolor_df.columns):
            best_efm_color[j, i, :] = colors.hex2color(hexcolor_df.at[g, o])

    axS12[0, 0].annotate('a', xy=(0.02, 0.98),
                         xycoords='axes fraction', ha='left', va='top',
                         size=20, color='white')
    axS12[0, 0].imshow(best_efm_color, interpolation='none', origin='lower')
    axS12[0, 0].set_xlabel(D.GLU_COL)
    axS12[0, 0].set_ylabel(D.OX_COL)
    axS12[0, 0].set_xticks(axis_params[D.GLU_COL]['xticks'])
    axS12[0, 0].set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
    axS12[0, 0].set_yticks(axis_params[D.OX_COL]['xticks'])
    axS12[0, 0].set_yticklabels(axis_params[D.OX_COL]['xticklabels'])

    # convert the standard glucose concentration to the imshow coordinates
    # we know that the minimum level is mapped to 0
    # and the maximum level is mapped to N
    # and that it is a logarithmic scale
    axS12[0, 0].plot([axis_params[D.GLU_COL]['std_ind'], axis_params[D.GLU_COL]['std_ind']],
                   [0, axis_params[D.OX_COL]['N']-1],
                   '--', color='grey', linewidth=1)
    axS12[0, 0].plot([0, axis_params[D.GLU_COL]['N']-1],
                   [axis_params[D.OX_COL]['std_ind'], axis_params[D.OX_COL]['std_ind']],
                   '--', color='grey', linewidth=1 )

    # mark the 3 selected EFMs in the Monod surface plot
    axS12[0, 0].annotate('max-gr', xy=(0.5, 0.8),
                       xycoords='axes fraction', ha='left', va='top',
                       size=14, color='k')
    axS12[0, 0].annotate('pareto', xy=(0.1, 0.4),
                       xycoords='axes fraction', ha='left', va='top',
                       size=14, color='k')
    axS12[0, 0].annotate('ana-lac', xy=(0.73, 0.1),
                       xycoords='axes fraction', ha='left', va='top',
                       size=14, color='k')
    axS12[0, 0].annotate('aero-ace', xy=(0.82, 0.29),
                       xycoords='axes fraction', ha='left', va='top',
                       size=14, color='k')

    best_efm_gr_df = monod_df.pivot(index=D.GLU_COL,
                                    columns=D.OX_COL,
                                    values=D.GROWTH_RATE_L)
    axS12[0, 1].annotate('b', xy=(0.02, 0.98),
                         xycoords='axes fraction', ha='left', va='top',
                         size=20, color='black')
    axS12[0, 1].set_xlabel(best_efm_gr_df.index.name)
    axS12[0, 1].set_xticks(axis_params[D.GLU_COL]['xticks'])
    axS12[0, 1].set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
    axS12[0, 1].get_yaxis().set_visible(False)
    pcol = axS12[0, 1].imshow(best_efm_gr_df.T, interpolation='spline16',
                              cmap='Oranges', vmin=0,
                              vmax=max_growth_rate, origin='lower')
    norm = colors.Normalize(vmin=0, vmax=max_growth_rate)
    colorbar.ColorbarBase(cbar_ax, cmap='Oranges', norm=norm)
    cbar_ax.set_title(D.GROWTH_RATE_L, loc='center')

    for i, efm in enumerate(monod_df['best_efm'].unique()):
        if efm in D.efm_dict:
            label = D.efm_dict[efm]['label']
        else:
            label = 'EFM %04d' % efm
        axS12[0, 2].plot([0, 1], [i, i],
                         label=label, color=D.efm_to_hex(efm), linewidth=3)
    axS12[0, 2].set_xlim(-1, 0)
    axS12[0, 2].set_ylim(-1, 0)
    axS12[0, 2].get_xaxis().set_visible(False)
    axS12[0, 2].get_yaxis().set_visible(False)
    axS12[0, 2].legend(fontsize=13, labelspacing=0.12, loc='center right')
    axS12[0, 2].axis('off')

    # make a Monod surface plot where certain features of the winning EFMs
    # are presented in color coding

    plot_parameters = [
        {'c': D.YIELD_L,    'cmap': 'magma_r', 'vmin': 0, 'vmax': 30 , 'ax': axS12[1, 0]},
        {'c': D.OXYGEN_L,   'cmap': 'magma_r', 'vmin': 0, 'vmax': 0.7, 'ax': axS12[1, 1]},
        {'c': D.ACE_L,      'cmap': 'magma_r', 'vmin': 0, 'vmax': 1.5, 'ax': axS12[1, 2]},
        {'c': D.LACTATE_L,  'cmap': 'magma_r', 'vmin': 0, 'vmax': 1.5, 'ax': axS12[2, 0]},
        {'c': D.ED_L,       'cmap': 'magma_r', 'vmin': 0, 'vmax': 2  , 'ax': axS12[2, 1]},
        {'c': D.PPP_L,      'cmap': 'magma_r', 'vmin': 0, 'vmax': 4.5, 'ax': axS12[2, 2]},
    ]
    pareto_data_df = figure_data['standard']

    for i, d in enumerate(plot_parameters):
        ax = d['ax']
        ax.annotate(chr(ord('a')+i+2), xy=(0.02, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    size=20, color='k')
        ax.set_title(d['c'])
        df = monod_df.join(pareto_data_df[d['c']], on='best_efm')
        df = df.pivot(index=D.GLU_COL,
                      columns=D.OX_COL,
                      values=d['c'])
        ax.set_xlabel(df.index.name)
        ax.set_ylabel(df.columns.name)
        pcol = ax.imshow(df.T, interpolation='none', cmap=d['cmap'],
                         origin='lower', vmin=d['vmin'], vmax=d['vmax'])
        pyplot.colorbar(pcol, ax=ax, shrink=0.6)

        # since the plot is made in a linear scale, we need to "manually" convert
        # the ticks to the log-scale using the index and columns of 'df'
        ax.set_xticks(axis_params[D.GLU_COL]['xticks'])
        ax.set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
        ax.set_yticks(axis_params[D.OX_COL]['xticks'])
        ax.set_yticklabels(axis_params[D.OX_COL]['xticklabels'])

    axS12[1, 1].get_yaxis().set_visible(False)
    axS12[1, 2].get_yaxis().set_visible(False)
    axS12[2, 1].get_yaxis().set_visible(False)
    axS12[2, 2].get_yaxis().set_visible(False)

    return figS12

def plot_conc_versus_uptake_figure(figure_data,
                                   sweep_cache_fname='sweep2d_win_200x200.csv'):
    """
        in order to compare to FBA predictions
        join the Monod surface data with the EFM rates table, in order to
        get specific rates for each winning EFM
    """
    monod_df, axis_params = get_monod_data(sweep_cache_fname)
    best_efm_hex = monod_df.pivot(index=D.GLU_COL,
                                  columns=D.OX_COL,
                                  values='hexcolor')
    best_efm_color = np.zeros((best_efm_hex.shape[1], best_efm_hex.shape[0], 3))
    for i, g in enumerate(best_efm_hex.index):
        for j, o in enumerate(best_efm_hex.columns):
            hexcolor = best_efm_hex.at[g, o]
            best_efm_color[j, i, :] = colors.hex2color(hexcolor)

    fig = pyplot.figure(figsize=(8, 8))
    ax_list = []

    ##################### Monod surface plot of winning EFMs ##################
    ax = fig.add_subplot(2, 2, 1)
    ax_list.append(ax)
    ax.imshow(best_efm_color, interpolation='none', origin='lower')
    ax.set_xticks(axis_params[D.GLU_COL]['xticks'])
    ax.set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
    ax.set_yticks(axis_params[D.OX_COL]['xticks'])
    ax.set_yticklabels(axis_params[D.OX_COL]['xticklabels'])
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(D.OX_COL)

    ################### growth rate surface plot vs concentrations ############
    ax = fig.add_subplot(2, 2, 2, projection='3d', facecolor='white')
    ax_list.append(ax)
    X = np.arange(0, axis_params[D.GLU_COL]['N'])
    Y = np.arange(0, axis_params[D.OX_COL]['N'])
    X, Y = np.meshgrid(X, Y)
    z_mat = monod_df.pivot(index=D.GLU_COL, columns=D.OX_COL, values=D.GROWTH_RATE_L).T.as_matrix()

    ax.plot_surface(X, Y, z_mat, facecolors=best_efm_color,
                    rstride=1, cstride=1,
                    antialiased=False, rasterized=True,
                    linewidth=0, shade=False)
    ax.plot_wireframe(X, Y, z_mat, rstride=6, cstride=6, linewidth=0.2,
                      edgecolor='k', alpha=0.3)
    ax.view_init(20, -120)
    ax.set_xticks(axis_params[D.GLU_COL]['xticks'])
    ax.set_xticklabels(axis_params[D.GLU_COL]['xticklabels'])
    ax.set_yticks(axis_params[D.OX_COL]['xticks'])
    ax.set_yticklabels(axis_params[D.OX_COL]['xticklabels'])
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(D.OX_COL)
    ax.set_zlabel(D.GROWTH_RATE_L, rotation=90)

    ###########################################################################
    OX_UPTAKE_L = 'O$_2$ uptake rate (a.u.)'
    GLU_UPRATE_L = 'glucose uptake rate (a.u.)'
    rates_df, _, _ = SweepInterpolator.get_general_params(min(ITER_RANGE))
    monod_df = monod_df.join(rates_df, on='best_efm')
    monod_df[OX_UPTAKE_L] = monod_df[D.R_OXYGEN_DEPENDENT].sum(1) * monod_df[D.GROWTH_RATE_L]
    monod_df[OX_UPTAKE_L] = monod_df[OX_UPTAKE_L].round(0)
    monod_df[GLU_UPRATE_L] = monod_df[D.R_GLUCOSE_IN] * monod_df[D.GROWTH_RATE_L]
    monod_df[GLU_UPRATE_L] = monod_df[GLU_UPRATE_L].round(0)
    monod_df[D.GROWTH_RATE_L] = monod_df[D.GROWTH_RATE_L].round(3)
    small_monod_df = monod_df[[GLU_UPRATE_L, OX_UPTAKE_L, D.GROWTH_RATE_L, 'hexcolor']].drop_duplicates()
    small_monod_df.sort_values(D.GROWTH_RATE_L, inplace=True)

    ########## 2D scatter plot of uptake rates (winning EFM as color) #########
    ax = fig.add_subplot(2, 2, 3)
    ax_list.append(ax)
    ax.scatter(x=small_monod_df[GLU_UPRATE_L],
               y=small_monod_df[OX_UPTAKE_L],
               s=15, c=small_monod_df['hexcolor'],
               linewidth=0)

    ax.set_xlabel(GLU_UPRATE_L)
    ax.set_ylabel(OX_UPTAKE_L)

    ############## 3D scatter plot of growth rate vs uptake rates #############

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax_list.append(ax)
    ax.scatter(xs=small_monod_df[GLU_UPRATE_L],
               ys=small_monod_df[OX_UPTAKE_L],
               zs=small_monod_df[D.GROWTH_RATE_L],
               s=15, c=small_monod_df['hexcolor'],
               cmap='Oranges', vmax=0.7, linewidth=0,
               alpha=1)

    ax.set_xlabel(GLU_UPRATE_L, labelpad=10)
    ax.set_ylabel(OX_UPTAKE_L, labelpad=10)
    ax.set_zlabel(D.GROWTH_RATE_L, labelpad=10)
    ax.view_init(20, -120)

    for i, ax in enumerate(ax_list):
        ax.annotate(chr(ord('a')+i), xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=20, ha='right', va='top')

    return fig

def plot_oxygen_sweep(ax, glucose_conc=None, N=200,
                      legend_loc='lower right', legend_fontsize=10):
    """make line plots of gr vs one of the axes (oxygen or glucose)"""
    if glucose_conc is None:
        glucose_conc = D.STD_CONC['glucoseExt']

    ox_grid = np.logspace(np.log10(D.MIN_CONC['oxygen']),
                          np.log10(D.MAX_CONC['oxygen']),
                          N)

    interp_data_df = pd.DataFrame(index=ox_grid, columns=D.efm_dict.keys())

    interpolator = SweepInterpolator.interpolate_2D_sweep(D.efm_dict.keys())
    for efm in interp_data_df.columns:
        interp_data_df[efm] = [interpolator.calc_gr(efm, glucose_conc, o)
                               for o in ox_grid]

    colors, labels = zip(*D.efm_dict.values())
    interp_data_df.plot(kind='line', ax=ax, linewidth=2, color=colors)

    ax.legend(labels,
              loc=legend_loc, fontsize=legend_fontsize, labelspacing=0.2)
    ax.set_xscale('log')
    ax.set_xlabel(D.OX_COL)
    ax.set_ylabel(r'growth rate [h$^{-1}$]')
    ax.set_ylim([0, None])

    # mark the line where 'standard' oxygen levels are
    std_ox = D.STD_CONC['oxygen']
    ax.plot([std_ox, std_ox], ax.get_ylim(), '--', color='grey', linewidth=1)
    ax.text(std_ox, ax.get_ylim()[1], '  std. $O_2$', ha='center', va='bottom',
            color='grey', fontsize=14)
    ax.text(0.02, 0.6, 'glucose (%d mM)' % glucose_conc, ha='left', va='center',
            rotation=90, fontsize=14, color='grey', transform=ax.transAxes)

def plot_glucose_sweep(ax, oxygen_conc=None, N=200, ylim=None,
                      legend_loc='upper left', legend_fontsize=10,
                      mark_glucose=True):
    """make line plots of gr vs one of the axes (oxygen or glucose)"""
    if oxygen_conc is None:
        oxygen_conc = D.STD_CONC['oxygen']

    glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                           np.log10(D.MAX_CONC['glucoseExt']),
                           N)

    interp_data_df = pd.DataFrame(index=glu_grid, columns=D.efm_dict.keys())

    interpolator = SweepInterpolator.interpolate_2D_sweep(D.efm_dict.keys())
    for efm in interp_data_df.columns:
        interp_data_df[efm] = [interpolator.calc_gr(efm, g, oxygen_conc)
                               for g in glu_grid]

    colors, labels = zip(*D.efm_dict.values())
    interp_data_df.plot(kind='line', ax=ax, linewidth=2, color=colors)

    if legend_loc is not None:
        ax.legend(labels,
                  loc=legend_loc, fontsize=legend_fontsize, labelspacing=0.2)
    else:
        ax.legend().remove()
    ax.set_xscale('log')
    ax.set_xlabel(D.GLU_COL)
    ax.set_ylabel(r'growth rate [h$^{-1}$]')
    if ylim is None:
        ax.set_ylim([0, None])
    else:
        ax.set_ylim(ylim)

    if mark_glucose:
        # mark the line where 'standard' oxygen levels are
        std_ox = D.STD_CONC['glucoseExt']
        ax.plot([std_ox, std_ox], ax.get_ylim(), '--', color='grey', linewidth=1)
        ax.text(std_ox, ax.get_ylim()[1], '  std. glucose', ha='center', va='bottom',
                color='grey', fontsize=14)
        ax.text(0.02, 0.6, '$O_2$ (%g mM)' % oxygen_conc, ha='left', va='center',
                rotation=90, fontsize=14, color='grey', transform=ax.transAxes)

def get_glucose_sweep_df(oxygen_conc=None, efm_list=None, N=200):
    
    if oxygen_conc is None:
        oxygen_conc = D.STD_CONC['oxygen']

    glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                           np.log10(D.MAX_CONC['glucoseExt']),
                           N)
    interpolator = SweepInterpolator.interpolate_2D_sweep(efm_list)

    interp_data_df = pd.DataFrame(index=glu_grid,
                                  columns=interpolator.efm_list)
    for efm in interpolator.efm_list:
        interp_data_df[efm] = [interpolator.calc_gr(efm, g, oxygen_conc)
                               for g in glu_grid]
    return interp_data_df

def get_anaerobic_glucose_sweep_df(figure_data, N=200):
    anaerobic_sweep_data_df = figure_data['monod_glucose_anae'].drop(9999)
    
    # filter all EMFs that have a > 1% drop in the function (it should be
    # completely monotonic, but some numerical errors should be okay).
    non_monotinic = (np.log(anaerobic_sweep_data_df).diff(axis=1) < 0)
    anaerobic_sweep_data_df[non_monotinic] = np.nan

    glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                           np.log10(D.MAX_CONC['glucoseExt']),
                           N)
    interp_df = anaerobic_sweep_data_df.transpose()
    interp_df = interp_df.append(
        pd.DataFrame(index=glu_grid, columns=anaerobic_sweep_data_df.index))
    interp_df = interp_df[~interp_df.index.duplicated(keep='first')]
    interp_df.sort_index(inplace=True)
    interp_df.index = np.log(interp_df.index)
    interpolated_df = interp_df.interpolate(method='polynomial', order=3)
    interpolated_df.index = np.exp(interpolated_df.index)
    return interpolated_df

def plot_oxygen_dual_pareto(data_df, ax, s=9,
                            std_ox=None, low_ox=None, std_glu=None,
                            draw_lines=True):
    std_ox = std_ox or D.STD_CONC['oxygen']
    low_ox = low_ox or D.LOW_CONC['oxygen']
    std_glu = std_glu or D.STD_CONC['glucoseExt']

    std_ox_df = pd.DataFrame(index=data_df.index,
                             columns=[D.GROWTH_RATE_L, D.YIELD_L])
    std_ox_df[D.YIELD_L] = data_df[D.YIELD_L]
    low_ox_df = pd.DataFrame(index=data_df.index,
                             columns=[D.GROWTH_RATE_L, D.YIELD_L])
    low_ox_df[D.YIELD_L] = data_df[D.YIELD_L]

    # calculate the growth rates in the lower oxygen level, using the
    # interpolated functions
    interpolator = SweepInterpolator.interpolate_2D_sweep()
    for efm in data_df.index:
        std_ox_df.at[efm, D.GROWTH_RATE_L] = \
            interpolator.calc_gr(efm, std_glu, std_ox)
        low_ox_df.at[efm, D.GROWTH_RATE_L] = \
            interpolator.calc_gr(efm, std_glu, low_ox)

    D.plot_dual_pareto(std_ox_df, 'std. O$_2$ (0.21 mM)',
                       low_ox_df, 'low O$_2$ (%g mM)' % low_ox,
                       s=s, ax=ax, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                       draw_lines=draw_lines)
    ax.set_xlim(-1e-3, None)
    ax.set_ylim(-1e-3, None)

def plot_glucose_dual_pareto(data_df, ax,
                             std_glu=None, low_glu=None, std_ox=None,
                             draw_lines=True):
    std_glu = std_glu or D.STD_CONC['glucoseExt']
    low_glu = low_glu or D.LOW_CONC['glucoseExt']
    std_ox = std_ox or D.STD_CONC['oxygen']

    std_glu_df = pd.DataFrame(index=data_df.index,
                              columns=[D.GROWTH_RATE_L, D.YIELD_L])
    std_glu_df[D.YIELD_L] = data_df[D.YIELD_L]
    low_glu_df = pd.DataFrame(index=data_df.index,
                              columns=[D.GROWTH_RATE_L, D.YIELD_L])
    low_glu_df[D.YIELD_L] = data_df[D.YIELD_L]

    # calculate the growth rates in the lower oxygen level, using the
    # interpolated functions
    interpolator = SweepInterpolator.interpolate_2D_sweep()
    for efm in data_df.index:
        std_glu_df.at[efm, D.GROWTH_RATE_L] = \
            interpolator.calc_gr(efm, std_glu, std_ox)
        low_glu_df.at[efm, D.GROWTH_RATE_L] = \
            interpolator.calc_gr(efm, low_glu, std_ox)

    D.plot_dual_pareto(std_glu_df, 'std. glucose (100 mM)',
                       low_glu_df, 'low glucose (%g mM)' % low_glu,
                       s=9, ax=ax, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                       draw_lines=draw_lines)
    ax.set_xlim(-1e-3, None)
    ax.set_ylim(-1e-3, None)

if __name__ == '__main__':
    figure_data = D.get_figure_data()
    rcParams['font.size'] = 12.0
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'
    rcParams['legend.fontsize'] = 'small'
    rcParams['axes.labelsize'] = 12.0
    rcParams['axes.titlesize'] = 12.0
    rcParams['xtick.labelsize'] = 10.0
    rcParams['ytick.labelsize'] = 10.0

    # run this script in order to calculate the extrapolated growth rates for
    # all the 200x200 grid and cache the results in a temp file for quick
    # access for the scripts that plot the data

    fig = pyplot.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    plot_oxygen_sweep(ax)

    ax = fig.add_subplot(1, 2, 2)
    plot_oxygen_dual_pareto(figure_data['standard'], ax)