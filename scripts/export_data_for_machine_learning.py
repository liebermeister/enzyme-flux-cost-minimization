# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:38:03 2015

@author: noore
"""

import zipfile, os
import pandas as pd
import definitions as D

TSNE_DIM_1 = 't-SNE dim1'
TSNE_DIM_2 = 't-SNE dim2'


if __name__ == '__main__':
    figure_data = D.get_figure_data()

    for fig_name in ['monod_glucose_aero', 'monod_glucose_anae']:
    
        zip_fname = D.DATA_FILES[fig_name][0][0]
        prefix, ext = os.path.splitext(os.path.basename(zip_fname))
        
        with zipfile.ZipFile(zip_fname, 'r') as z:
            rates_df = pd.read_csv(z.open('%s/rates.csv' % prefix, 'r'),
                                             header=0, index_col=0)
            stoich_df = pd.read_csv(z.open('%s/stoich.csv' % prefix, 'r'),
                                             header=None, index_col=None)
            kcat_df = pd.read_csv(z.open('%s/kcats.csv' % prefix, 'r'),
                                            header=None, index_col=None)
        
        #%%
        efms = list(rates_df.index)
        reactions = list(rates_df.columns)
        
        rates_df.index.name = 'efm'
        rates_df['efm'] = efms
        
        # convert rates_df to an SQL-style DataFrame, 
        # where the columns are 'efm', 'reaction', 'rate'
        # and remove all cases where the rate is 0
        melted_rates_df = pd.melt(rates_df, id_vars=['efm'])
        melted_rates_df.rename(columns={'variable': 'reaction', 'value': 'rate'}, inplace=True)
        melted_rates_df = melted_rates_df[melted_rates_df['rate'] != 0]
    
        # stoich_df is alread in SQL-style
        stoich_df.rename(columns={0: 'reaction', 1: 'metabolite', 2: 'coeff'}, inplace=True)
    
        # kcat_df is alread in SQL-style
        kcat_df.rename(columns={0: 'reaction', 1: 'kcat'}, inplace=True)
        kcat_df.set_index('reaction', inplace=True)
        
        # calculate degree of each metabolite using GROUP BY
        met_degree = stoich_df.groupby('metabolite').count()[['reaction']]
        met_degree.rename(columns={'reaction': 'degree'}, inplace=True)
        
        traindata_x = rates_df[reactions].copy()
        traindata_x.rename(columns=dict(zip(reactions, map(lambda s: s + ' rate', reactions))), inplace=True)
    
        # count the number of active reactions in each EFM
        traindata_x['# of reactions'] = melted_rates_df.groupby('efm').count()['reaction']
        traindata_x['sum of all rates'] = rates_df.sum(1)
        
        # calculate the number of metabolites participating in each EFM (i.e.
        # appearing as a substrate or product in at least one active reaction)
        efm_met_pairs = pd.merge(melted_rates_df, stoich_df, on='reaction')[['efm', 'metabolite']].drop_duplicates()
        traindata_x['# of metabolites'] = efm_met_pairs.groupby('efm').count()['metabolite']
    
        # count separately the active metabolites according to their degree
        efm_degree_pairs = efm_met_pairs.join(met_degree, on='metabolite')[['efm', 'degree']]
        traindata_x['# of metabolites with degree 1'] = efm_degree_pairs[efm_degree_pairs['degree'] == 1].groupby('efm').count()['degree']
        traindata_x['# of metabolites with degree 2'] = efm_degree_pairs[efm_degree_pairs['degree'] == 2].groupby('efm').count()['degree']
        traindata_x['# of metabolites with degree >=3'] = efm_degree_pairs[efm_degree_pairs['degree'] >= 3].groupby('efm').count()['degree']
    
        # add the rate/kcat of all reactions and the total
        rates_over_kcat = rates_df[reactions].divide(kcat_df.loc[reactions, 'kcat'].transpose())
        
        rates_over_kcat.rename(columns=dict(zip(reactions, map(lambda s: s + ' rate/kcat', reactions))), inplace=True)
        rates_over_kcat['total rate/kcat'] = rates_over_kcat.sum(1)
        
        traindata_x = traindata_x.join(rates_over_kcat)
        
        traindata_x.to_csv(os.path.join(D.OUTPUT_DIR, '%s_features.csv' % fig_name))
        
        #%%
        # prepare the data for learning, i.e. the enzyme costs of each EFM in several conditions
        # specifically in units of kD(protein) / (kD(biomass) / hour)
        traindata_y = figure_data[fig_name].reset_index()
        traindata_y = pd.melt(traindata_y, id_vars='efm', 
                              var_name='external glucose conc. [mM]',
                              value_name='growth rate [1/h]')
        traindata_y.sort_values(by=['efm', 'external glucose conc. [mM]'], inplace=True)
        traindata_y.to_csv(os.path.join(D.OUTPUT_DIR, '%s_enzyme_cost.csv' % fig_name), index=False)
    