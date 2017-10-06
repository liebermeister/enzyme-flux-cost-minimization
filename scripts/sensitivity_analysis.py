# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:44:19 2016

@author: noore

Calculate the local sensitivities of the enzyme cost to all kcats, Keq, and
Km values of all reaction or reaction/reactant pairs.
"""
import os, zipfile, re, sys, tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import definitions as D
from prepare_data import get_general_parameters_from_zipfile

class Sensitivity(object):

    def __init__(self, fig_name):
        zip_fnames, regex = D.DATA_FILES[fig_name]
        assert regex is None
        assert len(zip_fnames) == 1

        self.load_all_data(zip_fnames[0])

        self.efm_data_df = self.calculate_growth_rates()

        self.calculate_kcat_sensitivity()
        self.calculate_keq_sensitivity()
        self.calculate_km_sensitivity()

        self.kcat_sensitivity_df.to_csv(os.path.join(D.OUTPUT_DIR, 'sensitivity_kcat.csv'), float_format='%.2e')
        self.keq_sensitivity_df.to_csv(os.path.join(D.OUTPUT_DIR, 'sensitivity_keq.csv'), float_format='%.2e')
        self.km_sensitivity_df.to_csv(os.path.join(D.OUTPUT_DIR, 'sensitivity_km.csv'), float_format='%.2e')

        self.efm_data_df = pd.merge(self.efm_data_df,
                                    self.kcat_sensitivity_df, on=['efm', 'reaction'])
        self.efm_data_df = pd.merge(self.efm_data_df,
                                    self.keq_sensitivity_df, on=['efm', 'reaction'])

    def load_all_data(self, zip_fname):
        # load all data for all EFMs in standard conditions (i.e. both the
        # enzyme concentrations and the metabolite concentrations).
        prefix, ext = os.path.splitext(os.path.basename(zip_fname))

        with zipfile.ZipFile(zip_fname, 'r') as z:
            # read the stoichiometric data
            self.stoich_df = pd.DataFrame.from_csv(z.open('%s/stoich.csv' % prefix, 'r'),
                                                   header=None, index_col=None)
            self.stoich_df.rename(columns={0: 'reaction', 1: 'metabolite', 2: 'coefficient'}, inplace=True)
            self.stoich_df['reaction'] = self.stoich_df['reaction'].apply(D.FIX_REACTION_ID)
            # first read the table of reaction rates
            self.rates_df, self.params_df = get_general_parameters_from_zipfile(z, prefix)

            # go through all the files in the 'results' folder and read the enzyme data into
            # a dictionary from EFM to DataFrame
            csv_prefix = '^%s/results/enz-%s-r' % (prefix, prefix)
            fnames = [fname for fname in z.namelist() if re.search('%s(\d+)\.csv' % csv_prefix, fname)]
            fnames.sort()
            efms = map(lambda f: int(re.findall(csv_prefix + '(\d+)\.csv', f)[0]), fnames)

            frames = []
            for fname, efm in zip(fnames, efms):
                df = pd.DataFrame.from_csv(z.open(fname, 'r'), index_col=None)
                df.insert(0, 'efm', efm)
                frames.append(df)

            self.enz_df = pd.concat(frames, keys=None)
            self.enz_df.rename(columns={'r': 'reaction',
                                        'Val': 'enzyme_abundance'},
                                        inplace=True)
            self.enz_df['reaction'] = self.enz_df['reaction'].apply(D.FIX_REACTION_ID)

            # go through all the files in the 'results' folder and read the metabolite data into
            # a dictionary from EFM to DataFrame
            csv_prefix = '^%s/results/met-%s-r' % (prefix, prefix)
            fnames = [fname for fname in z.namelist() if re.search('%s(\d+)\.csv' % csv_prefix, fname)]
            fnames.sort()
            efms = map(lambda f: int(re.findall(csv_prefix + '(\d+)\.csv', f)[0]), fnames)

            frames = []
            for fname, efm in zip(fnames, efms):
                df = pd.DataFrame.from_csv(z.open(fname, 'r'), index_col=None)
                df.insert(0, 'efm', efm)
                frames.append(df)

            self.met_df = pd.concat(frames, keys=None)
            self.met_df.rename(columns={'m': 'metabolite',
                                        'Val': 'concentration'}, inplace=True)

            self.km_df = pd.DataFrame.from_csv(z.open('%s/kms.csv' % prefix, 'r'),
                                               header=None, index_col=None)
            self.km_df.rename(columns={0 : 'reaction', 1 : 'metabolite', 2 : 'Km'}, inplace=True)
            self.km_df['reaction'] = self.km_df['reaction'].apply(D.FIX_REACTION_ID)

    def calculate_growth_rates(self):
        # calculate the individual costs (enzyme abundance times its MW)
        enz_df = self.enz_df.join(self.params_df['weight'].fillna(0), on='reaction')
        enz_df['E_i'] = enz_df['enzyme_abundance'] * enz_df['weight']

        # calculate total enzyme costs and growth rate per EFM
        efms = enz_df['efm'].unique()
        v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * self.rates_df.loc[efms, D.R_BIOMASS]

        E_met = enz_df.groupby('efm').sum()['E_i']
        r_BM = (1.0 / E_met).mul(v_BM)
        r_BM.name = 'biomass_rate'  # r_BM (in units of 1/h)

        growth_rate = r_BM.apply(D.GR_FUNCTION)
        growth_rate.name = D.GROWTH_RATE_L

        yield_df = (D.BIOMASS_MW / D.C_IN_GLUCOSE) * \
                   (self.rates_df[D.R_BIOMASS] / self.rates_df[D.R_GLUCOSE_IN])
        yield_df.name = D.YIELD_L

        data_df = enz_df[['efm', 'reaction']]
        data_df.loc[:, 'enzyme_cost'] = enz_df['E_i'] / (D.BIOMASS_MW * D.SECONDS_IN_HOUR)

        # calculate the total weight of all the reactions per EFM
        enzyme_cost_sum_df = data_df.groupby('efm').sum()
        enzyme_cost_sum_df.rename(columns={'enzyme_cost': 'q'}, inplace=True)
        data_df = data_df.join(enzyme_cost_sum_df, on='efm')

        # collect all the information about each EFM and each reaction in one
        # big DataFrame
        data_df = data_df.join(r_BM, on='efm')
        data_df = data_df.join(growth_rate, on='efm')
        data_df = data_df.join(yield_df, on='efm')
        
        data_df.loc[:, 'dlnmu/dlnq'] = (D.GR_PARAM_B/D.GR_PARAM_A) * data_df[D.GROWTH_RATE_L] - 1.0
        data_df.loc[:, 'dmu/dq'] = data_df['dlnmu/dlnq'] * data_df[D.GROWTH_RATE_L] / data_df['q']
        
        data_df.sort_values(['efm', 'reaction'], inplace=True)
        return data_df

    def calculate_kcat_sensitivity(self):
        kcat_df = self.efm_data_df[['efm','reaction']].join(self.params_df['kcat'].fillna(0), on='reaction')
        kcat_df.loc[:, 'dq/dk'] = self.efm_data_df['enzyme_cost'] / kcat_df['kcat']
        kcat_df.loc[:, 'dlnq/dlnk'] = self.efm_data_df['enzyme_cost'] / self.efm_data_df['q']
        kcat_df.loc[:, 'dmu/dk'] = self.efm_data_df['dmu/dq'] * kcat_df['dq/dk']
        kcat_df.loc[:, 'dlnmu/dlnk'] = self.efm_data_df['dlnmu/dlnq'] * kcat_df['dlnq/dlnk']
        self.kcat_sensitivity_df = kcat_df

    def calculate_keq_sensitivity(self):
        """
            calculate the reaction quotient (Q) per reaction per EFM
            we use the formula:
                        log(Q_j) = sum_i (n_ij * log(c_i))
            where n_i are the stoichiometric coefficients and c_i are the concentrations
            in matrix notation it would be:
                        log(Q) = N' * log(C)
            where N' is the transposed stoichiometric matrix
        """
        thermo_df = pd.merge(self.stoich_df, self.met_df, on='metabolite')
        thermo_df.loc[:, 'n*log(c)'] = thermo_df['coefficient'] * np.log(thermo_df['concentration'])
        thermo_df = thermo_df[['efm','reaction','n*log(c)']].groupby(['efm', 'reaction']).sum()
        thermo_df.rename(columns={'n*log(c)':'Q'}, inplace=True)
        thermo_df.loc[:, 'Q'] = np.exp(thermo_df['Q'])
        thermo_df = thermo_df.reset_index()
        thermo_df = pd.merge(thermo_df, self.efm_data_df, on=['efm', 'reaction'])
        thermo_df = thermo_df.join(self.params_df['Keq'], on='reaction')

        thermo_df.loc[:, 'dq/dKeq'] = -thermo_df['enzyme_cost'] / (thermo_df['Keq'] *
                                (thermo_df['Keq']/thermo_df['Q'] - 1.0))

        thermo_df.loc[:, 'dlnq/dlnKeq'] = thermo_df['dq/dKeq'] * (thermo_df['Keq']/thermo_df['q'])
        thermo_df.fillna(0, inplace=True)
        thermo_df.loc[:, 'dmu/dKeq'] = thermo_df['dmu/dq'] * thermo_df['dq/dKeq']
        thermo_df.loc[:, 'dlnmu/dlnKeq'] = thermo_df['dlnmu/dlnq'] * thermo_df['dlnq/dlnKeq']

        # the sign of the Keq elasticity must be opposed to the sign of the
        # reaction rate, since when we increase Keq for a forward flowing reation
        # the enzyme cost must go down. For a backward flowing reaction,
        # increasing Keq will increase the cost.
        elast_df = thermo_df.pivot(index='efm', columns='reaction', values='dq/dKeq')
        assert ((self.rates_df * elast_df).fillna(0) <= 0).all().all()

        self.keq_sensitivity_df = thermo_df[['efm', 'reaction',
                                             'dq/dKeq', 'dlnq/dlnKeq',
                                             'dmu/dKeq', 'dlnmu/dlnKeq']]

    def calculate_km_sensitivity(self):
        """
            the partial derivative of the cost (q) as a function of a substrate's Km
            (K_s, for substrate 's') is given by:
                        dq/dK_s = (q / K_s) * n_s *
                                  [1 - eta_sat(c) * prod_i {K_i/c_i + 1} / (K_s/c_s + 1)]
            where eta_sat(c) is the saturation term, n_i is the stoichiometric coefficient
            of s in this reaction

            the partial derivative of the cost (q) as a function of product's Km
            (K_p, for product 'p') is given by:
                        dq/dK_p = -(q / K_p) * n_p * eta_sat(c) *
                                   prod_i {K_i/c_i} * prod_k {1 + c_j/K_j} / (K_p/c_p + 1)
        """

        sat_df = pd.merge(self.stoich_df, self.km_df, on=['reaction', 'metabolite'])
        sat_df = pd.merge(sat_df, self.met_df, on='metabolite')
        sat_df = pd.merge(sat_df, self.efm_data_df,
                          on=['efm', 'reaction'])
        sat_df.loc[:, 'c/Km'] = sat_df['concentration'] / sat_df['Km']

        sat_df.loc[:, '|n|*log(c/Km)']   = np.abs(sat_df['coefficient']) * np.log(sat_df['c/Km'])
        sat_df.loc[:, '|n|*log(1+c/Km)'] = np.abs(sat_df['coefficient']) * np.log(1.0 + sat_df['c/Km'])
        sat_df.loc[:, '|n|*log(Km/c+1)'] = np.abs(sat_df['coefficient']) * np.log(1.0/sat_df['c/Km'] + 1.0)

        subs_df = sat_df[sat_df['coefficient'] < 0]
        prod_df = sat_df[sat_df['coefficient'] > 0]

        subs_df = subs_df.groupby(['efm', 'reaction']).sum()
        prod_df = prod_df.groupby(['efm', 'reaction']).sum()

        subs_df.loc[:, 'prod(c_s/K_s)'] = np.exp(subs_df['|n|*log(c/Km)'])
        subs_df.loc[:, 'prod(1 + c_s/K_s)'] = np.exp(subs_df['|n|*log(1+c/Km)'])
        subs_df.loc[:, 'prod(K_s/c_s + 1)'] = np.exp(subs_df['|n|*log(Km/c+1)'])
        subs_df = subs_df.loc[:, ['prod(c_s/K_s)', 'prod(1 + c_s/K_s)', 'prod(K_s/c_s + 1)']]

        prod_df.loc[:, 'prod(c_p/K_p)'] = np.exp(prod_df['|n|*log(c/Km)'])
        prod_df.loc[:, 'prod(1 + c_p/K_p)'] = np.exp(prod_df['|n|*log(1+c/Km)'])
        prod_df.loc[:, 'prod(K_p/c_p + 1)'] = np.exp(prod_df['|n|*log(Km/c+1)'])
        prod_df = prod_df.loc[:, ['prod(c_p/K_p)', 'prod(1 + c_p/K_p)', 'prod(K_p/c_p + 1)']]

        sat_df = sat_df.join(subs_df, on=['efm', 'reaction'])
        sat_df = sat_df.join(prod_df, on=['efm', 'reaction'])

        sat_df.loc[:, 'eta_sat'] = sat_df['prod(c_s/K_s)'] / (sat_df['prod(1 + c_s/K_s)'] + sat_df['prod(1 + c_p/K_p)'] - 1)

        # now we need to calculate the product of all "other" substrates from |n|*log(1+c/Km)
        # besides the one with the changing Km. The easiest way would be to take the
        # full product, and divide by the metabolite term
        sat_df.loc[:, 'dq/dK_s'] = (sat_df['enzyme_cost'] / sat_df['Km']) * (
            1.0 - sat_df['eta_sat'] * sat_df['prod(K_s/c_s + 1)'] / (sat_df['Km'] / sat_df['concentration'] + 1))

        sat_df.loc[:, 'dq/dK_p'] = -(sat_df['enzyme_cost'] / sat_df['Km']) * sat_df['eta_sat'] * \
            (sat_df['prod(c_p/K_p)'] / sat_df['prod(c_s/K_s)']) * \
            (sat_df['prod(K_p/c_p + 1)'] / (sat_df['Km'] / sat_df['concentration'] + 1))

        # copy the sensitivities of the substrate KMs
        sat_df.loc[:, 'dq/dKm'] = sat_df['dq/dK_s']

        # override the values only for reaction/metabolite pairs which are of products
        sat_df.loc[sat_df['coefficient'] > 0, 'dq/dKm'] = sat_df.loc[sat_df['coefficient'] > 0, 'dq/dK_p']

        sat_df.loc[:, 'dlnq/dlnKm'] = sat_df['dq/dKm'] * (sat_df['Km']/sat_df['q'])
        
        sat_df.loc[:, 'dmu/dKm'] = sat_df['dmu/dq'] * sat_df['dq/dKm']
        sat_df.loc[:, 'dlnmu/dlnKm'] = sat_df['dlnmu/dlnq'] * sat_df['dlnq/dlnKm']
        
        self.km_sensitivity_df = sat_df[['efm', 'reaction', 'metabolite',
                                         'dq/dKm', 'dlnq/dlnKm',
                                         'dmu/dKm', 'dlnmu/dlnKm']]

    def read_sweep_enzyme_costs(self, prefix, regex):
        # read data from zip file (both enzyme and metabolite concentrations):
        zip_fname = os.path.join(D.DATA_DIR, '%s.zip' % prefix)

        with zipfile.ZipFile(zip_fname, 'r') as z:

            rates_df, params_df = get_general_parameters_from_zipfile(z, prefix)

            # go through all the files in the 'results' folder and read the enzyme data into
            # a dictionary from EFM to DataFrame
            csv_prefix = '^%s/results/enz-%s-r' % (prefix, prefix)
            fnames = [fname for fname in z.namelist() if re.search('%s(\d+)\.csv' % csv_prefix, fname)]
            fnames.sort()
            efms = map(lambda f: int(re.findall(csv_prefix + '(\d+)\.csv', f)[0]), fnames)

            frames = []
            for fname, efm in zip(fnames, efms):
                df = pd.DataFrame.from_csv(z.open(fname, 'r'), index_col=None)
                df.insert(0, 'efm', efm)
                frames.append(df)

            enz_df = pd.concat(frames, keys=None)

            param_list = map(lambda s: float(re.findall(regex + '([\d\.]+)', s)[0]), enz_df['Dim1'])
            enz_df.loc[:, 'Dim1'] = param_list
            enz_df.rename(columns={'Dim1': regex,
                                   'Dim2': 'reaction',
                                   'Val': 'enzyme_abundance'}, inplace=True)
            enz_df['reaction'] = enz_df['reaction'].apply(D.FIX_REACTION_ID)

        # calculate the cost in kD(protein) / (kD(biomass) / hour)
        # by dividing the total weight of protein per enzyme,
        # by how much gr/mole of biomass these enzyme can produce in 1 hour
        enz_df = enz_df.join(params_df, on='reaction')
        enz_df = enz_df.join(rates_df[D.R_BIOMASS], on='efm')
        enz_df.loc[:,'enzyme_cost'] = enz_df['enzyme_abundance'] * enz_df['weight'] / \
                                      (D.BIOMASS_MW * D.SECONDS_IN_HOUR * enz_df[D.R_BIOMASS])

        # sum the costs (group by EFM) and reshape the table so that
        # the index are EFMs and the columns are the value of the sweep parameter
        # value are the sum of enzyme costs
        enzyme_cost_sum_df = enz_df.groupby(['efm', regex], as_index=False).sum()
        enzyme_cost_sum_df = enzyme_cost_sum_df.pivot(index='efm',
                                                      columns=regex,
                                                      values='enzyme_cost')
        enzyme_cost_sum_df.index.name = 'efm'
        enzyme_cost_sum_df.columns.name = regex
        enzyme_cost_sum_df = enzyme_cost_sum_df.transpose()
        return enz_df, enzyme_cost_sum_df

    def verify_keq_sensitivity(self, prefix, regex, reaction):
        _, enzyme_cost_sum_df = self.read_sweep_enzyme_costs(prefix, regex)

        fit_df = self.keq_sensitivity_df[self.keq_sensitivity_df['reaction'] == reaction]
        fit_df.set_index('efm', inplace=True)
        fit_df.insert(len(fit_df.columns), 'slope', np.nan)

        for efm in enzyme_cost_sum_df.columns:
            z = np.polyfit(enzyme_cost_sum_df.index, enzyme_cost_sum_df[efm], deg=1)
            fit_df.at[efm, 'slope'] = z[0]

        return fit_df[~pd.isnull(fit_df['slope'])]

    def verify_km_sensitivity(self, prefix, regex, reaction, metabolite):
        _, enzyme_cost_sum_df = self.read_sweep_enzyme_costs(prefix, regex)

        fit_df = self.km_sensitivity_df[(self.km_sensitivity_df['reaction'] == reaction) & \
                                        (self.km_sensitivity_df['metabolite'] == metabolite)]
        fit_df.insert(len(fit_df.columns), 'slope', np.nan)
        fit_df.set_index('efm', inplace=True)

        for efm in enzyme_cost_sum_df.columns:
            z = np.polyfit(enzyme_cost_sum_df.index, enzyme_cost_sum_df[efm], deg=1)
            fit_df.at[efm, 'slope'] = z[0]

        return fit_df[~pd.isnull(fit_df['slope'])]

    def verifty_sensitivities(self):
        fig, axs = plt.subplots(3, 1, figsize=(4, 13))

        fit_keq = self.verify_keq_sensitivity(prefix='n38-p8',
                                              regex='keq-r2r-', reaction='R2r')
        fit_keq.plot(kind='scatter', x='slope', y='dq/dKeq', ax=axs[0], linewidth=0, marker='o')
        axs[0].set_ylabel(r'$\frac{dq}{dK_{eq}}$')
        axs[0].plot([-0.2, 0.2], [-0.2, 0.2], 'k-', alpha=0.2)

        fit_ks = self.verify_km_sensitivity(prefix='n38-p9',
                                            regex='kms-r10a.glu6p-', reaction='R10a', metabolite='glu6p')
        fit_ks.plot(kind='scatter', x='slope', y='dq/dKm', ax=axs[1], linewidth=0, marker='o')
        axs[1].set_ylabel(r'$\frac{dq}{dK_{s}}$')
        axs[1].plot([0, 0.15], [0, 0.15], 'k-', alpha=0.2)

        fit_kp = self.verify_km_sensitivity(prefix='n38-p10',
                                            regex='kms-r2r.fru6p-', reaction='R2r', metabolite='fru6p')
        fit_kp.plot(kind='scatter', x='slope', y='dq/dKm', ax=axs[2], linewidth=0, marker='o')
        axs[2].set_ylabel(r'$\frac{dq}{dK_{p}}$')
        axs[2].plot([-0.06, 0], [-0.06, 0], 'k-', alpha=0.2)

        fig.savefig(os.path.join(D.OUTPUT_DIR, 'sensitivity_verification.pdf'), dpi=300)

    def plot_sensitivity_cdfs(self):
        """
            plot the CDF of each type of sensitivity across all EFMs and reactions
        """
        def plot_cdf(df, col, ax, xscale='linear'):
            if xscale == 'log':
                ax.plot(df[col].abs().sort_values(), np.linspace(0.0, 1.0, num=df.shape[0]))
                ax.set_xlabel('abs(%s)' % col)
            else:
                ax.plot(df[col].sort_values(), np.linspace(0.0, 1.0, num=df.shape[0]))
                ax.set_xlabel(col)

            ax.set_xscale(xscale)
            ax.set_ylabel('cumulative distribution')
            ax.set_ylim(-0.01, 1.01)

        keq_tmp = self.keq_sensitivity_df[self.keq_sensitivity_df['dq/dKeq'] != 0]

        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
        plot_cdf(self.kcat_sensitivity_df, 'dmu/dk',       ax=axs[0,0], xscale='log')
        plot_cdf(keq_tmp,                  'dmu/dKeq',     ax=axs[0,1], xscale='log')
        plot_cdf(self.km_sensitivity_df,   'dmu/dKm',      ax=axs[0,2], xscale='log')
        plot_cdf(self.kcat_sensitivity_df, 'dlnmu/dlnk',   ax=axs[1,0], xscale='linear')
        plot_cdf(keq_tmp,                  'dlnmu/dlnKeq', ax=axs[1,1], xscale='linear')
        plot_cdf(self.km_sensitivity_df,   'dlnmu/dlnKm',  ax=axs[1,2], xscale='linear')
        fig.savefig(os.path.join(D.OUTPUT_DIR, 'sensitivity_cdf.pdf'), dpi=300)

    def plot_sensitivity_for_reaction(self, reaction):
        reaction_data_df = self.efm_data_df[self.efm_data_df['reaction'] == reaction]

        draw_keq_sensitivity = (reaction_data_df['dmu/dKeq'] != 0).any()

        substrates = self.stoich_df[(self.stoich_df['reaction'] == reaction) &
                                    (self.stoich_df['coefficient'] < 0)]['metabolite'].values
        products   = self.stoich_df[(self.stoich_df['reaction'] == reaction) &
                                    (self.stoich_df['coefficient'] > 0)]['metabolite'].values

        km_data = self.km_sensitivity_df[self.km_sensitivity_df['reaction'] == reaction]

        n_subfigs = 1 + len(substrates) + len(products)
        if draw_keq_sensitivity:
            n_subfigs += 1

        fig, axs = plt.subplots(1, n_subfigs, figsize=(4.5*n_subfigs, 3), sharey=True)
        axs_stack = list(axs)

        ax = axs_stack.pop(0)
        D.plot_basic_pareto(reaction_data_df, ax, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                            c='dlnmu/dlnk', cmap=D.pareto_cmap(0.83), linewidth=0)
        ax.set_title('sensitivity to $k_{cat}$ of %s' % reaction)
        ax.set_ylim(-1e-3, None)
        ax.set_xlim(-1e-3, None)

        if draw_keq_sensitivity:
            ax = axs_stack.pop(0)
            D.plot_basic_pareto(reaction_data_df, ax, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                                c='dlnmu/dlnKeq', cmap=D.pareto_cmap(0.11), linewidth=0)
            ax.set_title('sensitivity to $K_{eq}$ of %s' % reaction)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(-1e-3, 1.05*reaction_data_df[D.YIELD_L].max())
            ax.set_ylim(-1e-3, 1.05*reaction_data_df[D.GROWTH_RATE_L].max())

        for s in substrates:
            ax = axs_stack.pop(0)
            tmp_df = pd.merge(reaction_data_df, km_data[km_data['metabolite'] == s], on='efm')
            D.plot_basic_pareto(tmp_df, ax, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                                c='dlnmu/dlnKm', cmap=D.pareto_cmap(0.03), linewidth=0)
            ax.set_title('sensitivity to $K_S$ of %s : %s' % (reaction, s))
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(-1e-3, 1.05*reaction_data_df[D.YIELD_L].max())
            ax.set_ylim(-1e-3, 1.05*reaction_data_df[D.GROWTH_RATE_L].max())

        for p in products:
            ax = axs_stack.pop(0)
            D.plot_basic_pareto(tmp_df, ax, x=D.YIELD_L, y=D.GROWTH_RATE_L,
                                c='dlnmu/dlnKm', cmap=D.pareto_cmap(0.58), linewidth=0)
            ax.set_title('sensitivity to $K_P$ of %s : %s' % (reaction, p))
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(-1e-3, 1.05*reaction_data_df[D.YIELD_L].max())
            ax.set_ylim(-1e-3, 1.05*reaction_data_df[D.GROWTH_RATE_L].max())

        fig.savefig(os.path.join(D.OUTPUT_DIR, 'sensitivity_%s.pdf' % reaction), dpi=300)

    def plot_sensitivity_on_network(self, efm):
        """
            Use a colormap to overlay the sensitivity data on the network, for a
            specific EFM.
        """
        def plot_svg_network(data_dict, svg_layout):

            # first, we need to convert the reaction names that use lowercase 'r'
            # in the prefix to uppercase 'R'
            upper_prefix = lambda s : re.sub('(^r+)', lambda match: match.group(1).upper(), s)
            data_dict = dict(zip(map(upper_prefix, data_dict.keys()), data_dict.values()))

            vmod = vonda.PVisualizer(D.INPUT_SVG_FNAME, reaction_suffix='R',
                                     species_suffix='', colormap='custom.svg')
            with tempfile.NamedTemporaryFile(delete=True, suffix='.svg') as tmpfp:
                vmod.doMapReactions(data_dict, scaling_mode='linear',
                                    filename_out=tmpfp.name[:-4])
                svg_layout.add_figure(transform.fromstring(tmpfp.file.read()))


        sys.path.append(os.path.expanduser('~/git/VoNDA'))
        import vonda
        from svgutils import templates, transform

        slice_df = self.efm_data_df[self.efm_data_df['efm'] == efm].set_index('reaction').transpose()
        slice_km_df = self.km_sensitivity_df[self.km_sensitivity_df['efm'] == efm]

        subs_km_df = slice_km_df[slice_km_df['dq/dKm'] > 0].groupby('reaction').max().transpose()
        prod_km_df = slice_km_df[slice_km_df['dq/dKm'] < 0].groupby('reaction').min().transpose()

        layout = templates.ColumnLayout(1)
        plot_svg_network(slice_df.loc['dlnq/dlnk',:].to_dict(), layout)
        plot_svg_network(slice_df.loc['dlnq/dlnKeq',:].to_dict(), layout)
        plot_svg_network(subs_km_df.loc['dlnq/dlnKm',:].to_dict(), layout)
        plot_svg_network((-prod_km_df.loc['dlnq/dlnKm',:]).to_dict(), layout)
        layout.save(os.path.join(D.OUTPUT_DIR, 'sensitivity_efm%03d.pdf' % efm))

    def plot_sensitivity_as_errorbar(self, ax, reaction, foldchange=2):
        """
            we plot the 'error' for the growth rate, for a multiplicative
            change in the k_cat of a single reaction
        """

        ax.set_title('sensitivity of growth rate to a %g-fold '
                     'change in $k_{cat}$ of %s' %
                      (foldchange, reaction))
        ax.set_xlabel(D.YIELD_L)

        reaction_data_df = self.efm_data_df.groupby('efm').first()
        reaction_data_df = reaction_data_df[[D.YIELD_L, D.GROWTH_RATE_L]].reset_index()
        _tmp_df = self.efm_data_df[self.efm_data_df['reaction'] == reaction]
        _tmp_df = _tmp_df[['efm', 'dlnmu/dlnk']].set_index('efm')
        reaction_data_df = reaction_data_df.join(_tmp_df, on='efm', how='left')
        
        # to calculate the relative error, we use the following
        # mu / mu_0 = (k / k_0) ^ (d ln mu / d ln k)
        

        # plot all the 0-sensitivity EFMs first with grey points
        reaction_data_df[pd.isnull(reaction_data_df['dlnmu/dlnk'])].plot(
            kind='scatter', x=D.YIELD_L, y=D.GROWTH_RATE_L,
            color=(0.7, 0.3, 0.5), alpha=1, ax=ax, s=4)
        
        # plot all the sensitive EFMs using an error bar plot
        err_df = reaction_data_df[~pd.isnull(reaction_data_df['dlnmu/dlnk'])]
        yerr_up = err_df[D.GROWTH_RATE_L] * (foldchange**(-err_df['dlnmu/dlnk']) - 1.0)
        yerr_down = -err_df[D.GROWTH_RATE_L] * (foldchange**err_df['dlnmu/dlnk'] - 1.0)
        yerr = np.vstack([yerr_up.values, yerr_down.values])
        ax.errorbar(err_df[D.YIELD_L],
                    err_df[D.GROWTH_RATE_L],
                    yerr=yerr,
                    fmt='.', capsize=3, capthick=0.5,
                    elinewidth=0.5, color=(0.7, 0.3, 0.5))

################################################################################

if __name__ == '__main__':

    s = Sensitivity('standard')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    s.plot_sensitivity_as_errorbar(ax[0], 'R80', foldchange=2)
    s.plot_sensitivity_as_errorbar(ax[1], 'R6r', foldchange=2)
    fig.savefig(os.path.join(D.OUTPUT_DIR, 'sensitivity_errorbars.pdf'))

    sys.exit(0)
    #%% cumulative distribution plots
    s.plot_sensitivity_cdfs()
    #s.verifty_sensitivities()

    #%% plot the sensitivity of all EFMs to

    #%% EFM Pareto plots (gr vs yield) where sensitivities are color coded
    s.plot_sensitivity_for_reaction('R6r')
    s.plot_sensitivity_for_reaction('R54ra')
    s.plot_sensitivity_for_reaction('R1')
    s.plot_sensitivity_for_reaction('R80')
    s.plot_sensitivity_for_reaction('R22')

    #%% network plots with sensitivites as edge colors
    #svg_text = s.plot_sensitivity_on_network(5)
