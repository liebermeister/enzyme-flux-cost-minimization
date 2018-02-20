# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:01:45 2015

@author: eladn
"""

import pandas as pd
import pulp
import os
import definitions as D
from prepare_data import get_concatenated_raw_data
import numpy as np
import matplotlib

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

class FluxProjection(object):

    def __init__(self):
        self.reaction_name_df = pd.read_csv('../data/reaction_name_mapping.csv',
                                            header=0, index_col=0)
        self.reaction_name_df.index = map(D.FIX_REACTION_ID, self.reaction_name_df.index)
        
        self.gerosa_df = pd.read_csv('../data/fluxes_from_gerosa_et_al.csv',
                                     header=0, index_col=0)
        self.gerosa_df.index = map(lambda r: D.FIX_REACTION_ID(r, throw_exception=False),
                                   self.gerosa_df.index)
        
        self.bounds_df = pd.read_csv('../data/bounds.csv',
                                     header=0, index_col=0)
        self.bounds_df.index = map(lambda r: D.FIX_REACTION_ID(r, throw_exception=False),
                                   self.bounds_df.index)
        
        self.stoich_df = pd.read_csv('../data/stoich.csv',
                                     header=0, index_col=0).transpose()
        self.stoich_df.columns = map(lambda r: D.FIX_REACTION_ID(r, throw_exception=False),
                                     self.stoich_df.columns)
        self.stoich_df.columns.name = 'reaction_id'

        self.project_fluxes()

    def project_fluxes(self):
        flux_means = self.gerosa_df.iloc[:, 0]
        flux_stderr = self.gerosa_df.iloc[:, 1]

        fluxes_df = pd.DataFrame(index=self.stoich_df.columns)
        fluxes_df.loc[flux_means.index, D.MEAS_FLUX_L] = flux_means
        fluxes_df.loc[flux_means.index, D.MEAS_STDEV_L] = flux_stderr

        lp = pulp.LpProblem("FLUX_L1", pulp.LpMinimize)

        all_reactions = list(self.stoich_df.columns)
        all_metabolites = list(self.stoich_df.index)
        self.measured_reactions = list(flux_means.index)

        v_pred = pulp.LpVariable.dicts('v_pred', all_reactions)
        v_meas = pulp.LpVariable.dicts('v_meas', self.measured_reactions)
        v_resid = pulp.LpVariable.dicts('residual', self.measured_reactions)

        # add flux bounds
        for i in all_reactions:
            lp += (v_pred[i] >= self.bounds_df.loc[i, 'lower_bound']), 'lower_bound_%s' % i
            lp += (v_pred[i] <= self.bounds_df.loc[i, 'upper_bound']), 'upper_bound_%s' % i

        # add constraint for each measured reaction i:
        # |v_meas[i] - flux_means[i]| <= flux_stderr[i]
        # v_resid[i] >= |v_pred[i] - v_meas[i]|
        for i in self.measured_reactions:
            lp += (v_meas[i] <= flux_means[i] + flux_stderr[i]), 'measured_upper_%s' % i
            lp += (v_meas[i] >= flux_means[i] - flux_stderr[i]), 'measured_lower_%s' % i
            lp += (v_pred[i] - v_resid[i] <= v_meas[i]), 'abs_diff_upper_%s' % i
            lp += (-v_pred[i] - v_resid[i] <= -v_meas[i]), 'abs_diff_lower_%s' % i

        # also set the objective to be minimizing sum_i abs_diff[i]
        objective = pulp.lpSum(v_resid.values())
        lp.setObjective(objective)

        # add stoichiometric constraints for all internal metabolites: S_int * v = 0
        for m in all_metabolites:
            row = [self.stoich_df.loc[m, i] * v_pred[i] for i in all_reactions]
            lp += (pulp.lpSum(row) == 0), 'mass_balance_%s' % m

        lp.solve()
        #lp.writeLP("tmp/flux_mapping.lp")

        fluxes_df.loc[all_reactions, D.PRED_FLUX_L] = \
            list(map(lambda i: pulp.value(v_pred[i]), all_reactions))
        fluxes_df.loc[self.measured_reactions, D.RESID_L] = \
            list(map(lambda i: pulp.value(v_resid[i]), self.measured_reactions))
        fluxes_df /= pulp.value(v_pred['R70']) # normalize all fluxes to the biomass flux (i.e. set it to 1)

        self.fluxes_df = fluxes_df.reset_index().join(self.reaction_name_df,
                                                      on='reaction_id')
        
        noname = self.fluxes_df[pd.isnull(self.fluxes_df.name)].index
        self.fluxes_df.loc[noname, 'name'] = \
            self.fluxes_df.loc[noname, 'reaction_id']

    def plot_flux_projection_results(self):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        fig.subplots_adjust(wspace=0.5)
        axs[0].plot([-400, 400], [-400, 400], 'k', alpha=0.3, linewidth=0.5)
        self.fluxes_df.plot(kind='scatter', x=D.MEAS_FLUX_L, y=D.PRED_FLUX_L,
                       xerr=D.MEAS_STDEV_L, ax=axs[0], linewidth=0, s=10,
                       color=(0.7,0.2,0.5))
        for i, row in self.fluxes_df.iterrows():
            xy = row[[D.MEAS_FLUX_L, D.PRED_FLUX_L]]
            axs[0].annotate(row['name'], xy, xytext=(10,-5), textcoords='offset points',
                            family='sans-serif', fontsize=10, color='darkslategrey')

        _df = self.fluxes_df.loc[~pd.isnull(self.fluxes_df[D.RESID_L]), :]
        _df = _df.set_index('name')
        _df[D.RESID_L].plot(kind='bar', ax=axs[1], color=(0.7, 0.2, 0.5))
        axs[1].set_xlabel('residual [mM/s]')

        fig.savefig(os.path.join(D.OUTPUT_DIR, 'flux_projection.pdf'))

        self.fluxes_df.to_pickle(os.path.join(D.TEMP_DIR, 'measured_fluxes.pkl'))
        self.fluxes_df.to_csv(os.path.join(D.TEMP_DIR, 'measured_fluxes.csv'))

    def plot_correlations(self):
        # Figure that calculates the Euclidean distance between each EFM and
        # the "experimental" flow, and overlays that information on the
        # standard "Pareto" plot

        exp_flux_df = self.fluxes_df.copy()
        
        # remove the exchange reactions (xchg_*)
        exp_flux_df = exp_flux_df.loc[exp_flux_df.reaction_id.str.find('xchg') != 0, :]
        exp_flux_df.reaction_id = exp_flux_df.reaction_id.apply(D.FIX_REACTION_ID)

        fig0, axs0 = plt.subplots(1, 2, figsize=(15, 7))
        rates_df, params_df, km_df, enzyme_abundance_df = \
            get_concatenated_raw_data('standard')

        CORR_FLUX_L = 'correlation with exp fluxes'
        LOG_LIKELIHOOD_L = 'log likelihood of flow'

        figure_data = D.get_figure_data()
        data = figure_data['standard']

        data[CORR_FLUX_L] = rates_df.transpose().corr().loc[9999]
        # calculate the likelihood of each EFM according to the measured flux
        # distribution
        data[LOG_LIKELIHOOD_L] = 0

        joined_rates = rates_df.T
        joined_rates['std'] = exp_flux_df[D.MEAS_STDEV_L]
        joined_rates['std'] = joined_rates['std'].fillna(0) + 1.0 # add a baseline stdev of 10%
        for efm in data.index:
            x = (joined_rates[efm] - joined_rates[9999]) / joined_rates['std']
            log_likelihood = -(x**2).sum()/2
            data.loc[efm, LOG_LIKELIHOOD_L] = log_likelihood

        data.loc[data[D.STRICTLY_ANAEROBIC_L], D.GROWTH_RATE_L] = 0 # remove oxygen-sensitive EFMs
        cmap = D.pareto_cmap(0.88)
        D.plot_basic_pareto(data, axs0[0], x=D.YIELD_L, y=D.GROWTH_RATE_L,
                            c=CORR_FLUX_L, cmap=cmap, vmin=0, vmax=1, linewidth=0, s=20)
        D.plot_basic_pareto(data, axs0[1], x=D.YIELD_L, y=D.GROWTH_RATE_L,
                            c=LOG_LIKELIHOOD_L, cmap=cmap, linewidth=0, s=20,
                            vmin=-100000, vmax=0)

        for ax in axs0:
            for efm in D.efm_dict.keys():
                xy = np.array(data.loc[efm, [D.YIELD_L, D.GROWTH_RATE_L]].tolist())
                xytext = xy + np.array((-1, 0.025))
                ax.annotate(xy=xy, s=D.efm_dict[efm]['label'],
                            xycoords='data', xytext=xytext,
                            arrowprops=dict(facecolor='black',
                            shrink=0.05, width=2, headwidth=4))
            ax.set_xlim(-1e-3, 1.1*data[D.YIELD_L].max())
            ax.set_ylim(-1e-3, 1.15*data[D.GROWTH_RATE_L].max())
        axs0[0].set_title('distance from measured fluxes (correlation)')
        axs0[1].set_title('distance from measured fluxes (likelihood)')
        fig0.tight_layout()

        fig0.savefig(os.path.join(D.OUTPUT_DIR, 'Fig_flux_correlation.pdf'))

if __name__ == '__main__':
    f = FluxProjection()
    f.plot_flux_projection_results()
    f.plot_correlations()
