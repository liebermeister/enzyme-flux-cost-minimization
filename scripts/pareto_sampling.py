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

PICKLE_FNAME = os.path.join(D.TEMP_DIR, 'pareto_sampling.pkl')

def write_cache_files():
    data_list = []
    for set_fname in ['n39-p1', 'n39-p80', 'n39-p81', 'n39-p82']:
        zip_fname = os.path.join(D.DATA_DIR, set_fname + '.zip')
        data_list.append(read_pareto_zipfile(zip_fname))
    data = pd.concat(data_list, axis=0, join='inner')
    # the indexes of these DataFrames are not EFMs as usually happens in
    # other datasets, but just arbitraty sequence numbers. To avoid 
    # duplicates, we override them with new unique indexes
    data.index = range(data.shape[0])
    data.to_pickle(PICKLE_FNAME)

if __name__ == '__main__':
    write_cache_files()