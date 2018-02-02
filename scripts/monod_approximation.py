"""
Created on Sat Dec  3 17:32:45 2016

@author: eladn
"""
import matplotlib.pyplot as plt
import definitions as D
from sensitivity_analysis import Sensitivity
import numpy as np
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

figure_data = D.get_figure_data()

with PdfPages(os.path.join(D.OUTPUT_DIR, 'monod_approximation.pdf')) as pdf:
    for efm, (color, label) in D.efm_dict.items():
        rxn = 'R1' # the glucose transporter
        met = 'glucoseExt'
        
        monod_df = figure_data['monod_glucose_aero'].loc[efm, :].reset_index()
        monod_df.columns = [D.GLU_COL, D.GROWTH_RATE_L]
        
        GLU_CONC = ['100.', '10.', '1.', '0.1', '0.01', '0.001']
        color_cycle = sns.color_palette(n_colors=len(GLU_CONC))
        
        monod_curves = {}
        
        for i, c in enumerate(GLU_CONC):
            d = {}
            d['c0'] = float(c)
            s = Sensitivity(D.DATA_FILES['monod_glucose_aero'][0][0],
                            'mext-%s-%s' % (met, c))
            
            _df1 = s.efm_data_df.set_index(['efm', 'reaction'])
            d['q0'] = _df1.at[(efm, rxn), 'q']
            
            _df2 = s.km_sensitivity_df.set_index(['efm', 'reaction', 'metabolite'])
            dlnq_dlnC = -_df2.at[(efm, rxn, met), 'dlnq/dlnKm']
            
            d['Km'] = -d['c0'] / (1.0 + 1.0 / dlnq_dlnC)
            d['qmin'] = d['q0'] / (1 + d['Km']/d['c0'])
            d['color'] = color_cycle[i]
            monod_curves[c] = d
            
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        for ax, max_glu in zip(axs.flat, [0.02, 0.4, 8]):
            glu_grid = np.linspace(0, max_glu, 200)
            ax.set_xlim(0, max_glu)
            ax.set_xlabel('External glucose [mM]')
            ax.set_ylabel(D.GROWTH_RATE_L)
            ax.plot(monod_df[D.GLU_COL], monod_df[D.GROWTH_RATE_L], '-',
                    color=(0.7, 0.7, 0.7), linewidth=4)
            for d in monod_curves.values():
                mu = lambda c : D.GR_FUNCTION(c / (d['qmin'] * (c + d['Km'])))
                ax.plot(glu_grid, list(map(mu, glu_grid)), '-', color=d['color'], alpha=0.7)
                ax.plot(d['c0'], D.GR_FUNCTION(1/d['q0']), 'o', color=d['color'])
        
        ax = axs[1, 1]
        glu_grid = np.logspace(np.log10(D.MIN_CONC['glucoseExt']),
                               np.log10(D.MAX_CONC['glucoseExt']), 200)
        ax.plot(monod_df[D.GLU_COL], monod_df[D.GROWTH_RATE_L], '-',
                color=(0.7, 0.7, 0.7), linewidth=4)
        for d in monod_curves.values():
            q = lambda c : d['qmin'] * (1.0 + d['Km']/c)
            mu = lambda c : D.GR_FUNCTION(1.0 / q(c))
            ax.plot(glu_grid, list(map(mu, glu_grid)), '-', color=d['color'], alpha=0.7)
            ax.plot(d['c0'], D.GR_FUNCTION(1/d['q0']), 'o', color=d['color'])
        ax.set_xscale('log')
        ax.set_xlabel('External glucose [mM]')
        ax.set_ylabel(D.GROWTH_RATE_L)
        fig.suptitle(label + ' [O$_2$] = 0.21 mM')
        fig.tight_layout()
        
        pdf.savefig(fig)