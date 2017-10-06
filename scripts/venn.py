# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:17:11 2016

@author: noore
"""

import pandas as pd
import os
import definitions as D
from prepare_data import get_concatenated_raw_data

def count_efms():
    rates1_df, _, _ = get_concatenated_raw_data('standard')
    rates2_df, _, _ = get_concatenated_raw_data('anaerobic')
    rates_df = pd.concat([rates1_df, rates2_df]).drop_duplicates()
    
    produces_biomass = (rates_df[D.R_BIOMASS] > 1e-8)
    requires_oxygen = (rates_df[D.R_OXYGEN_DEPENDENT] != 0).any(axis=1)
    sensitive_oxygen = (rates_df[D.R_OXYGEN_SENSITIVE].abs() > 1e-8).any(axis=1)
    
    print 'produces biomass = %d' % produces_biomass.sum()
    print 'requires oxygen = %d' % (produces_biomass & (requires_oxygen == True) & (sensitive_oxygen == False)).sum()
    print 'sensitive to oxygen = %d' % (produces_biomass & (requires_oxygen == False) & (sensitive_oxygen == True)).sum()
    
    print 'aerobic = %d' % (produces_biomass & (sensitive_oxygen == False)).sum()
    print 'anaerobic = %d' % (produces_biomass & (requires_oxygen == False)).sum()
    print 'infeasible = %d' % (produces_biomass & (requires_oxygen == True) & (sensitive_oxygen == True)).sum()
    print 'facultative = %d' % (produces_biomass & (requires_oxygen == False) & (sensitive_oxygen == False)).sum()
    
    # write the data table of the main Pareto figure to a csv file:
    data = pd.read_pickle(os.path.join(D.TEMP_DIR, 'standard.pkl'))
    data.to_csv(os.path.join(D.TEMP_DIR, 'standard.csv'))
    
    data = pd.read_pickle(os.path.join(D.TEMP_DIR, 'anaerobic.pkl'))
    data.to_csv(os.path.join(D.TEMP_DIR, 'anaerobic.csv'))
    
    
if __name__ == '__main__':
    count_efms()