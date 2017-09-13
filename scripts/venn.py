# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:17:11 2016

@author: noore
"""

import pandas as pd
import os
import definitions as D

efms = pd.DataFrame.from_csv(os.path.join(D.DATA_DIR, 'efms26082016.csv'))

produces_biomass = (efms[D.R_BIOMASS] > 1e-8)
requires_oxygen = (efms[D.R_OXYGEN_IN].abs() > 1e-8).any(axis=1)
sensitive_oxygen = (efms[D.R_PFL] > 1e-8)

print 'produces biomass = %d' % produces_biomass.sum()
print 'requires oxygen = %d' % (produces_biomass & (requires_oxygen == True) & (sensitive_oxygen == False)).sum()
print 'sensitive to oxygen = %d' % (produces_biomass & (requires_oxygen == False) & (sensitive_oxygen == True)).sum()
print 'infeasible = %d' % (produces_biomass & (requires_oxygen == True) & (sensitive_oxygen == True)).sum()
print 'facultative = %d' % (produces_biomass & (requires_oxygen == False) & (sensitive_oxygen == False)).sum()

print 'aerobic = %d' % (produces_biomass & (requires_oxygen == True) & (sensitive_oxygen == True)).sum()
print 'anaerobic = %d' % (produces_biomass & (requires_oxygen == False) & (sensitive_oxygen == False)).sum()


# write the data table of the main Pareto figure to a csv file:
data = pd.read_pickle(os.path.join(D.TEMP_DIR, 'standard.pkl'))
data.to_csv(os.path.join(D.TEMP_DIR, 'standard.csv'))

data = pd.read_pickle(os.path.join(D.TEMP_DIR, 'anaerobic.pkl'))
data.to_csv(os.path.join(D.TEMP_DIR, 'anaerobic.csv'))