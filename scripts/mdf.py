#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:41:29 2018

@author: noore
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from os import path

import sys
sys.path.append(path.expanduser('~/git/equilibrator-api'))
from equilibrator_api.reaction import Reaction
from equilibrator_api.pathway import Pathway
from equilibrator_api.bounds import Bounds

from prepare_data import get_concatenated_raw_data
import definitions as D
figure_data = D.get_figure_data()

def read_stoichiometry():
    """
        read the stoichiometric matrix from Excel file, map metabolites to KEGG,
        discard external reactions and balance reactions
        (with Pi and H2O, which are not in the original model).
    """
    # First, we need to map the stoichiometric matrix to KEGG ID format
    # in order to use with the MDF method from equilibrator-api
    
    stoich_df = pd.read_excel(D.ECOLI_MODEL_FNAME,
                              sheet_name='Stoichiometry Matrix')
    
    # external metabolite list
    ext_mets = ['BIOMASS', 'ATP_main'] + stoich_df.columns[stoich_df.columns.str.match('.+_ext')].tolist()
    
    # drop all reactions that involve external metabolites
    stoich_df = stoich_df[ (stoich_df[ext_mets] == 0).all(axis=1) ]
    
    # and then drop the external metabolite columns
    stoich_df = stoich_df.drop(ext_mets, axis=1)
    
    
    # transpose S and map metabolites to KEGG IDs
    met2kegg = pd.read_excel(D.ECOLI_MODEL_FNAME, sheet_name='Metabolites',
                             index_col=1)['Compound ID']
    stoich_df.rename(columns=met2kegg.to_dict(), inplace=True)
    
    # H2O is also missing in the model, but we have to use chemical balancing to
    # find out its stoichiometry values
    stoich_df['C00009'] = 0
    stoich_df['C00001'] = 0
    
    for rxn, row in stoich_df.iterrows():
        r = Reaction(row.to_dict())
        while not r.check_full_reaction_balancing():
            imbalance = r._get_reaction_atom_balance()
            if 'P' in imbalance:
                row['C00009'] = -imbalance['P']
            elif 'O' in imbalance:
                row['C00001'] = -imbalance['O']
            else:
                print(imbalance)
                raise Exception('cannot balance reaction "%s": %s' % (rxn, r.write_formula()))
            r = Reaction(row.to_dict())
    
    # transpose S to comply with the standard orientation for constraint-based flux models
    return stoich_df.transpose()

if __name__ == '__main__':
    S = read_stoichiometry()
    rxn_df = S.apply(lambda r: Reaction(r.to_dict())).to_frame()
    rxn_df.columns = ['Reaction']
    rxn_df['dG0_prime'] = list(rxn_df['Reaction'].apply(lambda r: r.dG0_prime()))
    
    rates_df, _, _, _ = get_concatenated_raw_data('standard')
    
    # keep only reactions that are in S (i.e., not external)
    rates_df = rates_df[S.columns]
    
    #%%
    BOUNDS = Bounds(default_lb=1e-6,
                    default_ub=1e-2)
    mdf_data_dict = {}
    for efm, row in rates_df.iterrows():
        fluxes = row[row != 0]
        pathway = Pathway(rxn_df.loc[fluxes.index, 'Reaction'],
                          fluxes.values,
                          rxn_df.loc[fluxes.index, 'dG0_prime'],
                          bounds=BOUNDS)
        mdf_data_dict[efm] = pathway.calc_mdf()
        print(efm, mdf_data_dict[efm].mdf)

    #%%
    with PdfPages(path.join(D.OUTPUT_DIR, 'mdf_corr.pdf')) as pdf:
    
        data = figure_data['standard']
        # remove oxygen-sensitive EFMs
        data = data[~data[D.STRICTLY_ANAEROBIC_L]]
        data['MDF'] = [mdf_data_dict[efm].mdf for efm in data.index]
    
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        data.plot.scatter(x=D.YIELD_L, y=D.GROWTH_RATE_L, c='MDF', ax=axs[0], cmap='viridis')
        data.plot.scatter(x='MDF', y=D.GROWTH_RATE_L, c=D.YIELD_L, ax=axs[1], cmap='viridis')
        fig.suptitle(r'Bounds: %.3g $<$ [c] $<$ %.3g'
                     % (BOUNDS.default_lb, BOUNDS.default_ub))
        
        pdf.savefig(fig)
    
        # plot MDF profiles for the focal EFMs
        for efm, (col, lab) in D.efm_dict.items():
            mdf_data = mdf_data_dict[efm]
            fig1 = mdf_data.conc_plot
            fig1.suptitle(lab)
            fig1.get_axes()[0].set_xlim(1e-10, None)
            fig1.tight_layout()
            pdf.savefig(fig1)
            pdf.savefig(mdf_data.mdf_plot)
    