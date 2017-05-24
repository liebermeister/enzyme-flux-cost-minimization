# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 2015

@author: noore
"""

import re, os, zipfile
import numpy as np
import pandas as pd
import definitions as D

def get_general_parameters_from_zipfile(z, prefix):
    rates_df = pd.DataFrame.from_csv(z.open('%s/rates.csv' % prefix, 'r'),
                                     header=0, index_col=0)
    rates_df.index.name = 'efm'
    rates_df.columns.name = 'reaction'

    kcats_df = pd.DataFrame.from_csv(z.open('%s/kcats.csv' % prefix, 'r'),
                                     header=None, index_col=0)
    kcats_df.rename(columns={1: 'kcat'}, inplace=True)
    kcats_df.index.name = 'reaction'

    keqs_df = pd.DataFrame.from_csv(z.open('%s/keqs.csv' % prefix, 'r'),
                                     header=None, index_col=0)
    keqs_df.rename(columns={1: 'Keq'}, inplace=True)
    keqs_df.index.name = 'reaction'

    weights_df = pd.DataFrame.from_csv(z.open('%s/weights.csv' % prefix, 'r'),
                                     header=None, index_col=0)
    weights_df.rename(columns={1: 'weight'}, inplace=True)
    weights_df.index.name = 'reaction'

    # create a DataFrame that contains all reaction parameters (i.e. kcats and weights)
    params_df = kcats_df.join(keqs_df).join(weights_df)

    params_df.index = map(D.FIX_REACTION_ID, params_df.index)
    rates_df.columns = map(D.FIX_REACTION_ID, rates_df.columns)

    return rates_df, params_df

def get_df_from_sweep_zipfile(zip_fname, regex=None):
    """
        Read all the enzyme information from the zip file and return a single
        DataFrame with four columns:
            efm, sweep_param, reaction, enzyme_conc
    """
    prefix, ext = os.path.splitext(os.path.basename(zip_fname))

    with zipfile.ZipFile(zip_fname, 'r') as z:
        # first read the table of reaction rates
        rates_df, params_df = get_general_parameters_from_zipfile(z, prefix)

        # go through all the files in the 'results' folder and read the data into
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

        full_df = pd.concat(frames, keys=efms)

    full_df = full_df.rename(columns={'Dim2': 'reaction', 'Val': 'enzyme_abundance'})
    full_df['reaction'] = full_df['reaction'].apply(D.FIX_REACTION_ID)

    if regex is not None:
        param_list = []
        for s in full_df['Dim1']:
            regex_match = re.findall(regex + '([\d\.]+)', s)
            if regex_match == []:
                raise ValueError('The string "%s" was not found in the first '
                                 'column of the result files: %s' % (regex, s))
            param_list += [float(regex_match[0])]
        full_df['Dim1'] = param_list
        full_df = full_df.rename(columns={'Dim1': regex})

    # join the abundance DataFrame with the parameters, and multiply each
    # enzyme abundance by its weight
    full_df = full_df.join(params_df, on='reaction')
    full_df['E_i'] = full_df['enzyme_abundance'] * full_df['weight'].fillna(0)

    return rates_df, full_df

def get_biomass_rates_from_sweep_zipfile(zip_fname, regex):
    rates_df, full_df = get_df_from_sweep_zipfile(zip_fname, regex)

    efms = full_df['efm'].unique()
    v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * rates_df.loc[efms, D.R_BIOMASS]

    # sum up the indivitdual enzyme costs (E_i) per EFM and condition to
    # get the value of E_met
    E_met = full_df.groupby(['efm', regex]).sum()['E_i'].reset_index()
    E_met = E_met.pivot(index=regex, columns='efm', values='E_i')
    r_BM = (1.0 / E_met).mul(v_BM)

    r_BM.index.name = regex
    r_BM.columns.name = 'efm'
    return r_BM.transpose()

def read_sweep_zipfile(zip_fname, regex):
    return get_biomass_rates_from_sweep_zipfile(zip_fname, regex).applymap(D.GR_FUNCTION)

def get_df_from_pareto_zipfile(zip_fname):
    prefix, ext = os.path.splitext(os.path.basename(zip_fname))

    with zipfile.ZipFile(zip_fname, 'r') as z:
        rates_df, params_df = get_general_parameters_from_zipfile(z, prefix)

        # go through all the files in the 'results' folder and read the data into
        # a single dataframe with reaction IDs as the index and EFMs as
        # as the columns
        csv_prefix = '^%s/results/enz-%s-r' % (prefix, prefix)
        fnames = [fname for fname in z.namelist() if re.search('%s(\d+)\.csv' % csv_prefix, fname)]
        efms = map(lambda f: int(re.findall(csv_prefix + '(\d+)\.csv', f)[0]), fnames)
        enzyme_abundance_df = pd.DataFrame(index=params_df.index, columns=sorted(efms), dtype=float)
        for fname, efm in zip(fnames, efms):
            tmp_df = pd.DataFrame.from_csv(z.open(fname, 'r'), index_col=0)
            tmp_df.index = map(D.FIX_REACTION_ID, tmp_df.index)
            enzyme_abundance_df[[efm]] = tmp_df
        enzyme_abundance_df.columns.name = 'efm'
        enzyme_abundance_df.index.name = 'reaction'

    enzyme_abundance_df = enzyme_abundance_df.fillna(0).transpose()

    return rates_df, params_df, enzyme_abundance_df

def read_pareto_zipfile(zip_fname):
    """
        Read all data from the zip file containing eMCM analysis results of all
        EFMs and prepare a matrix that will be used for Pareto plots.
    """
    rates_df, params_df, enzyme_abundance_df = get_df_from_pareto_zipfile(zip_fname)

    # the enzyme cost (E_met) is a DataFrame with reactions as index and EFMs as
    # columns. Values are the mass of the enzyme [in grams] dedicated for that reaction
    # in each EFM (i.e. the abundances multiplied by the molecular masses)
    E_i = enzyme_abundance_df.mul(params_df['weight'].fillna(0))
    E_met = E_i.sum(1) # total metabolic enzyme in grams per EFM

    # calculate the biomass flux in grams per hour
    v_BM = D.BIOMASS_MW * D.SECONDS_IN_HOUR * rates_df[D.R_BIOMASS] # in [gCDW/h]

    # calculate the biomass production per enzyme investment
    r_BM = v_BM / E_met # in [1/h]

    uptake_rate = D.C_IN_GLUCOSE * rates_df[D.R_GLUCOSE_IN] # in mol C

    # calculate the yield of every EFM using the rates DataFrame
    data = pd.DataFrame(index=rates_df.index, dtype=float)
    data.index.name = 'EFM'

    # calculate interesting flux ratios

    # Yield in g [dw] per mol C glucose
    data[D.GLUCOER_UPTAKE_L] = rates_df[D.R_GLUCOSE_IN]
    data[D.YIELD_L]      = (D.BIOMASS_MW / D.C_IN_GLUCOSE) * (rates_df[D.R_BIOMASS] / rates_df[D.R_GLUCOSE_IN])

    # Yield in mol C biomass per mol C glucose
    data[D.YIELD_MOL_L]  = (D.C_IN_BIOMASS / D.C_IN_GLUCOSE) * (rates_df[D.R_BIOMASS] / rates_df[D.R_GLUCOSE_IN])
    data[D.ACE_L]        = D.C_IN_ACETATE * rates_df[D.R_ACETATE_OUT] / uptake_rate
    data[D.SUCCINATE_L]  = D.C_IN_SUCCINATE * rates_df[D.R_SUCCINATE_OUT] / uptake_rate
    data[D.LACTATE_L]    = D.C_IN_LACTATE * rates_df[D.R_LACTATE_OUT] / uptake_rate
    data[D.FORMATE_L]    = D.C_IN_FORMATE * rates_df[D.R_FORMATE_OUT] / uptake_rate
    data[D.NH3_L]        = rates_df[D.R_NH3_IN] / uptake_rate
    data[D.OXYGEN_L]     = 0.5 * rates_df[D.R_OXYGEN_IN].sum(1) / uptake_rate
    data[D.MAITENANCE_L] = rates_df[D.R_MAINTENANCE]
    data[D.PPP_L]        = rates_df[D.R_PPP] / rates_df[D.R_GLUCOSE_IN]
    data[D.TCA_L]        = rates_df[D.R_TCA] / rates_df[D.R_GLUCOSE_IN]
    data[D.ED_L]         = rates_df[D.R_ED]  / rates_df[D.R_GLUCOSE_IN]
    data[D.UPPER_GLYCOLYSIS_L] = rates_df[D.R_UPPER_GLYCOLYSIS] / rates_df[D.R_GLUCOSE_IN]
    data[D.PDH_L]        = rates_df[D.R_PDH] / rates_df[D.R_GLUCOSE_IN]

    # calculate the different cost and benefit measures
    data[D.N_REACTION_L] = (rates_df.abs() > 1e-8).sum(1)
    data[D.TOT_FLUX_L] = (np.abs(rates_df)).sum(1) / rates_df[D.R_GLUCOSE_IN]

    # calculate the sum of fluxes divided by SA, i.e. the precursor
    # for calculating specific activity

    # first, divide each abs(rate) by the specific activity of the enzyme
    specific_activity =  params_df['kcat'] / params_df['weight']
    v_over_sa = np.abs(rates_df).multiply(1.0 / specific_activity, axis='columns')

    # then, normalize each flux by the biomass flux [in gCDW/h]
    sum_v_over_sa = v_over_sa.sum(1) / v_BM

    data[D.TOT_FLUX_SA_L] = sum_v_over_sa
    data[D.INV_TOT_FLUX_SA_L] = 1.0 / sum_v_over_sa
    data[D.BIOMASS_PROD_PER_ENZ_L] = r_BM
    data[D.TOT_ENZYME_L] = 1.0 / r_BM
    data[D.GROWTH_RATE_L] = r_BM.apply(D.GR_FUNCTION)

    data[D.STRICTLY_ANAEROBIC_L] = (rates_df[D.R_PFL] > 1e-8)
    data[D.STRICTLY_AEROBIC_L] = (rates_df[D.R_OXYGEN_IN].abs() > 1e-8).any(1)
    data[D.SUC_FUM_CYCLE_L] = rates_df.loc[:, D.R_SUC_FUM_CYCLE].min(1)

    return data

def get_concatenated_data_from_zipfiles(fig_name):
    zip_fnames, regex = D.DATA_FILES[fig_name]
    data_list = []
    for zip_fname in zip_fnames:
        if regex is None:
            data_list.append(read_pareto_zipfile(zip_fname))
        else:
            data_list.append(read_sweep_zipfile(zip_fname, regex))
    data = pd.concat(data_list, axis=0, join='inner')
    return data

def get_concatenated_raw_data(fig_name):
    zip_fnames, regex = D.DATA_FILES[fig_name]

    if regex is None: # this is a "pareto" type zip file
        data_list = map(get_df_from_pareto_zipfile, zip_fnames)
        rates_df_list, params_df_list, enz_df_list = zip(*data_list)

        rates_df = pd.concat(rates_df_list, axis=0, join='inner')
        enzyme_abundance_df = pd.concat(enz_df_list, axis=0, join='inner')
        params_df = params_df_list[0]
        return rates_df, params_df, enzyme_abundance_df
    else:  # this is a "sweep" type zip file
        data_list = map(lambda f: get_df_from_sweep_zipfile(f, regex), zip_fnames)
        rates_df_list, full_df_list = zip(*data_list)
        rates_df = pd.concat(rates_df_list, axis=0, join='inner')
        full_df = pd.concat(full_df_list, axis=0, join='inner')
        return rates_df, full_df

if __name__ == '__main__':
    for fig_name in D.DATA_FILES.keys():
        data = get_concatenated_data_from_zipfiles(fig_name)
        grouped = data.groupby(level=0)
        unique_data = grouped.last()
        unique_data.to_pickle(os.path.join(D.TEMP_DIR, fig_name + '.pkl'))

    #from phase_surface_plots import write_cache_files
    write_cache_files()
