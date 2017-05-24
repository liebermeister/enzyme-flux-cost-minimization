# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:10:01 2015

@author: noore
"""
import sys, re, os, zipfile, tempfile
from definitions import INPUT_SVG_FNAME, ZIP_SVG_FNAME
import pandas as pd
import definitions as D

sys.path.append(os.path.expanduser('~/git/VoNDA'))
import vonda

###############################################################################

#%%
# gather the EFMs from both aerobic and anaerobic result files
# map their numbers to the original full list values
rates_dfs = []
for fig_name in ['standard', 'anaerobic']:
    zip_fname = D.DATA_FILES[fig_name][0][0]
    with zipfile.ZipFile(zip_fname, 'r') as z:
        prefix, ext = os.path.splitext(os.path.basename(zip_fname))
        rates_df = pd.DataFrame.from_csv(z.open('%s/rates.csv' % prefix, 'r'),
                                         header=0, index_col=0)
        rates_df.rename(index=D.DATA_FILES[fig_name][2], inplace=True)
        rates_dfs.append(rates_df)

# drop duplicates (i.e. EFMs that can operate both aerobically and
# anaerobically)
rates_df = pd.concat(rates_dfs).sort_index().drop_duplicates()

# we need to rename all the reaction names in rates_df to have an uppercase 'R'
# instead of lowercase as the reaction prefix
upper_prefix = lambda s : (s, re.sub('(^r+)', lambda match: match.group(1).upper(), s))
rates_df.rename(columns=dict(map(upper_prefix, rates_df.columns)),
                inplace=True)
##%%
#
#vmod = vonda.PVisualizer(INPUT_SVG_FNAME, reaction_suffix='R',
#                         species_suffix='', colormap='magma.svg')
#vmod.doMapReactions(rates_df.loc[9999,:].to_dict(), scaling_mode='linear',
#                    filename_out='example',
#                    old_width=1.0, max_width=4.0)
#sys.exit(0)

#%%
with zipfile.ZipFile(ZIP_SVG_FNAME, 'w') as z:

    vmod = vonda.PVisualizer(INPUT_SVG_FNAME, reaction_suffix='R',
                             species_suffix='', colormap='magma.svg')
    with tempfile.NamedTemporaryFile(delete=True, suffix='.svg') as tmpfp:
        vmod.doMapOfReactionIDs(filename_out=tmpfp.name[:-4])
        fp = z.write(tmpfp.name, arcname='reaction_ids.svg')

    for efm, fluxes in rates_df.iterrows():
        with tempfile.NamedTemporaryFile(delete=True, suffix='.svg') as tmpfp:
            vmod.doMapReactions(fluxes.to_dict(), scaling_mode='linear',
                                filename_out=tmpfp.name[:-4],
                                old_width=1.0, max_width=4.0)
            fp = z.write(tmpfp.name, arcname='efm%04d.svg' % efm)
