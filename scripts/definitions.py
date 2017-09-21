# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:08:08 2015

@author: noore
"""
import os, re, csv
import pandas as pd
import matplotlib
import seaborn as sns
from colorsys import hls_to_rgb
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import definitions as D
from hashlib import sha1
from matplotlib import rcParams

sns.set()
sns.set(style="white", context="paper", font="monospaced")

rcParams['font.size'] = 14.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['legend.fontsize'] = 'medium'
rcParams['axes.labelsize'] = 14.0
rcParams['axes.titlesize'] = 14.0
rcParams['xtick.labelsize'] = 12.0
rcParams['ytick.labelsize'] = 12.0


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{txfonts},\usepackage{lmodern},\usepackage{cmbright}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['font.weight'] = 'medium'
matplotlib.rcParams['font.style'] = 'normal'
matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

BASE_DIR = os.path.expanduser('~/git/flux-enzyme-cost-minimization')
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMP_DIR = os.path.join(BASE_DIR, 'tmp')
OUTPUT_DIR = os.path.join(BASE_DIR, 'res')
ZIP_SVG_FNAME = os.path.join(OUTPUT_DIR, 'all_efms.zip')
INPUT_SVG_FNAME = os.path.join(DATA_DIR, 'Ecoli_Carlson_2016_05_09.svg')
REACTION_NAME_FNAME = os.path.join(DATA_DIR, 'reaction_name_mapping.csv')
PROTEOME_FNAME = os.path.join(DATA_DIR, 'protein_abundance_from_schmidt_et_al.csv')

DATA_FILES = {
         'standard':             [['n39-p1'],  None],
         'anaerobic':            [['n39-p7'],  None],
         'low_kcat_r6r':         [['n39-p14'], None],
         'sweep_oxygen':         [['n39-p11'], 'mext-oxygen-'],
         'sweep_glucose':        [['n39-p16'], 'mext-glucoseExt-'],
         'sweep_kcat_r6r':       [['n39-p13'], 'kcat-r6r-'],
         'sweep_kcat_r70':       [['n39-p15'], 'kcat-r70-'],
         'monod_glucose_aero':   [['n39-p17'], 'mext-glucoseExt-'],
         'monod_glucose_anae':   [['n39-p18'], 'mext-glucoseExt-'],
        }

for v in DATA_FILES.values():
    v[0] = [os.path.join(DATA_DIR, f + '.zip') for f in v[0]]

SELECT_EFM_DF = pd.DataFrame.from_csv(os.path.join(DATA_DIR, 'n39-select_EFMs.csv'))

efm_dict = OrderedDict()
for efm in SELECT_EFM_DF.index:
    efm_dict[int(efm)] = SELECT_EFM_DF.loc[efm, ['color', 'label']]

R_BIOMASS = 'R70'
R_GLUCOSE_IN = 'R1'
R_ACETATE_OUT = 'R91'
R_FORMATE_OUT = 'R96'
R_SUCCINATE_OUT = 'R95'
R_LACTATE_OUT = 'R94'
R_NH3_IN = 'R93'
R_OXYGEN_DEPENDENT = ['R80', 'R27'] # oxphos and sdh
#R_OXYGEN_SENSITIVE = ['R20', 'R27b'] # pfl and frd
R_OXYGEN_SENSITIVE = ['R20'] # pfl (without frd)
R_MAINTENANCE = 'R82'
R_PPP = 'R10a'
R_TCA = 'R22'
R_ED = 'R60'
R_PDH = 'R21'
R_UPPER_GLYCOLYSIS = 'R5r'
R_SUC_FUM_CYCLE = ['R27', 'R27b']

C_IN_BIOMASS = 580
BIOMASS_MW   = 20666 # [mg/mmol]
C_IN_GLUCOSE = 6.0
C_IN_ACETATE = 2.0
C_IN_FORMATE = 1.0
C_IN_ACETATE = 2.0
C_IN_SUCCINATE = 4.0
C_IN_LACTATE = 3.0
SECONDS_IN_HOUR = 3600.0

# settings for sweeps and 2D phase plots
STD_CONC = {'glucoseExt': 1e2, 'oxygen': 2.1e-1} # in mM
MIN_CONC = {'glucoseExt': 1e-4, 'oxygen': 1e-3}  # in mM
MAX_CONC = {'glucoseExt': 1e4, 'oxygen': 1e1}    # in mM
LOW_CONC = {'glucoseExt': 1e-1, 'oxygen': 2.1e-3} # in mM
GLU_COL = 'glucose level (mM)'
OX_COL = 'O$_2$ level (mM)'

ALPHA_CCM = 0.25 # The mass fraction of metabolic enzyme within the proteome
ALPHA_PROT = 0.5 # The mass fraction of protein within biomass
GR_PARAM_A = 0.27 # unitless
GR_PARAM_B = 0.2 # in [h]
# linear growth rate function:
#GR_FUNCTION = lambda r_BM : r_BM * ALPHA_CCM * ALPHA_PROT

# non-linear growth rate function:
GR_FUNCTION = lambda r_BM : r_BM * GR_PARAM_A * ALPHA_PROT * \
                            (1.0 + GR_PARAM_B * ALPHA_PROT * r_BM)**(-1)

GENERAL_CMAP = 'magma_r'
GR_HEATMAP_CMAP = 'magma_r'
EPISTATIS_CMAP = 'RdBu'

def FIX_REACTION_ID(r):
    hits = re.findall('(^[Rr]+)(\d+.*)', r)
    if len(hits) != 1:
        raise ValueError('reaction name does not match the required pattern: ' + r)
    return hits[0][0].upper() + hits[0][1].lower()

with open(REACTION_NAME_FNAME, 'r') as fp:
    REACTION_DICT = {r['id']: r['name'] for r in csv.DictReader(fp)}

# read the protein abundance measurements and
# convert from copies/cell to mM
SINGLE_CELL_VOLUME = 2.4e-15 # Liters, Schmidt et al. 2015 (for glucose chemostat mu=0.35)
AVOGADRO = 6.02e23
copies_per_cell_to_mM = lambda x: float(x) / SINGLE_CELL_VOLUME / AVOGADRO * 1.0e3
with open(PROTEOME_FNAME, 'r') as fp:
    PROTEOME_DICT = {r['id']: copies_per_cell_to_mM(r['abundance'])
                     for r in csv.DictReader(fp)}

def GET_REACTION_NAME(r):
    if r == 'other':
        return 'other'
    return REACTION_DICT[FIX_REACTION_ID(r)]


GLUCOER_UPTAKE_L = r'glucose uptake [mM / s]'
GROWTH_RATE_L = r'growth rate [h$^{-1}$]'
BIOMASS_PROD_PER_ENZ_L = 'biomass production [gr dw h$^{-1}$ / gr enz]'
YIELD_L =     'biomass yield [gr dw / mol C glc]'
YIELD_MOL_L = 'biomass yield [mol C biomass / mol C glc]'
ACE_L =       'acetate secretion [mol C ace / mol C glc]'
FORMATE_L =   'formate secretion [mol for / mol C glc]'
SUCCINATE_L = 'succinate secretion [mol C suc / mol C glc]'
LACTATE_L =   'lactate secretion [mol C lac / mol C glc]'
NH3_L =       'ammonia uptake [mol NH3 / mol C glc]'
PPP_L =       'pentose phosphate flux (relative to uptake)'
TCA_L =       'TCA cycle flux (citrate synthase relative to uptake)'
ED_L  =       'ED pathway flux (relative to uptake)'
UPPER_GLYCOLYSIS_L = 'upper glycolysis (relative to uptake)'
PDH_L =       'pyruvate dehydrogenase (relative to uptake)'
MAITENANCE_L = 'maintenance cost [mM ATP / s]'
STRICTLY_AEROBIC_L = 'aerobic'
STRICTLY_ANAEROBIC_L = 'anaerobic'
SUC_FUM_CYCLE_L = 'succinate-fumarate cycle'


COST_UNITS = '[gr enz / gr dw h$^{-1}$]'
OXYGEN_L = r'oxygen uptake [mol O$_2$ / mol C glc]'
N_REACTION_L = 'number of active reactions'
TOT_FLUX_L = 'sum of fluxes relative to uptake [a.u.]'
TOT_FLUX_SA_L = r'$\Sigma_i \frac{|v_i| \cdot w_i}{k_{\mathrm{cat},i}}$ %s' % COST_UNITS
INV_TOT_FLUX_SA_L = r'Pathway specific activity'
TOT_ENZYME_L = 'total enzyme %s' % COST_UNITS

## labels for measured flux data
MEAS_FLUX_L = 'measured fluxes from Gerosa et al. [mM/s]'
MEAS_STDEV_L = 'standard deviation [mM/s]'
PRED_FLUX_L = 'projected fluxes [mM/s]'
RESID_L = 'residual [mM/s]'

## colors for plots:
PARETO_NEUTRAL_COLOR = (0.9, 0.7, 0.7)
PARETO_STRONG_COLOR = (0.4, 0.2, 0.9)
BAR_COLOR = (0.8, 0.4, 0.5)
PARETO_CMAP_LOWEST = (0.8, 0.8, 0.8)
PARETO_CMAP_HIGHEST = (0.1, 0.1, 0.1)

# H values between 0.165 and 0.5 are banned (too green)
# hls_to_rgb(0.165, 0.5, 1) -> (1.0, 1.0, 0.0)
# hls_to_rgb(0.50, 0.5, 1)  -> (0.0, 1.0, 1.0)

GAP_START = 0.165
GAP_SIZE = 0.335
rand2hue = lambda x: x*(1.0-GAP_SIZE) + GAP_SIZE*(x > GAP_START/(1.0-GAP_SIZE))
REMAINDER_L = 'other'
ANAEROBIC_OXYGEN_LEVEL = 1e-2 # in mM, i.e. equal to 10 uM
PREDIFINED_COLORS = {REMAINDER_L: hls_to_rgb(0, 0.8, 0),
                     'R80': hls_to_rgb(0.0, 0.5, 0.8), # red - ox phos
                     'R1':  hls_to_rgb(0.6, 0.5, 0.7), # blue - glucose uptake
                     'RR9': hls_to_rgb(0.1, 0.5, 0.7)} # yellow - PEP->PYR

def get_figure_data():
    fig_names = DATA_FILES.keys()
    fig_dfs = map(lambda s: pd.read_pickle(os.path.join(TEMP_DIR, s + '.pkl')),
                  fig_names)
    return dict(zip(fig_names, fig_dfs))

def get_projected_exp_fluxes():
    fluxes_df = pd.read_pickle(os.path.join(TEMP_DIR, 'measured_fluxes.pkl'))
    return fluxes_df

def pareto_cmap(h_mid):
    """
        Creates a colormap where the edges are light and dark grey, and the
        center is the provided HUSL color
    """
    rgb_mid = hls_to_rgb(h_mid, 0.5, 0.6)
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        'bright-color-dark', [PARETO_CMAP_LOWEST, rgb_mid, PARETO_CMAP_HIGHEST])

def plot_basic_pareto(data, ax, x, y, s=10, marker='o', c=None,
                      facecolors=(0.85, 0.85, 0.85), edgecolors='none',
                      paretofacecolors='none', paretoedgecolors='none',
                      paretosize=20, paretomarker='s',
                      efm_dict=None,
                      show_efm_labels=True,
                      **kwargs):
    """
        make plot gr vs yield for all EFMs
    """
    xdata = data.loc[:, x]
    ydata = data.loc[:, y]
    if c is not None:
        # if the c-value of all the data points is 0, use gray color
        # (otherwise, by default, the cmap will give 0 the middle color)
        cdata = data.loc[:, c]
        CS = ax.scatter(xdata, ydata, s=s, c=cdata, marker=marker,
                        facecolors=facecolors, edgecolors=edgecolors,
                        **kwargs)
        cbar = plt.colorbar(CS, ax=ax)
        cbar.set_label(c)
    else:
        cdata = None
        CS = ax.scatter(xdata, ydata, s=s, marker=marker,
                        facecolors=facecolors, edgecolors=edgecolors,
                        **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if efm_dict is not None:
        for efm, (col, lab) in efm_dict.iteritems():
            if efm in data.index:
                ax.plot(data.at[efm, x], data.at[efm, y], markersize=5,
                        marker=marker, color=col, label=None)
                if show_efm_labels:
                    ax.annotate(lab, xy=(data.at[efm, x], data.at[efm, y]),
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', va='bottom', color=col)

    if paretofacecolors != 'none' or paretoedgecolors != 'none':
        # find the EFMs which are on the pareto front and mark them
        pareto_xy = []
        for i in ydata.sort_values(ascending=False).index:
            if pareto_xy == [] or data[x][i] > pareto_xy[-1][0]:
                pareto_xy.append((data[x][i], data[y][i]))

        xpareto, ypareto = zip(*pareto_xy)
        ax.scatter(xpareto, ypareto, s=paretosize, marker=paretomarker,
                   facecolors=paretofacecolors, edgecolors=paretoedgecolors)
    return CS

def plot_dual_pareto(data0, label0, data1, label1, ax, x, y,
                     s=15, marker='o', c0=None, c1=None, draw_lines=True,
                     **kwargs):
    """
        Plot a comparative Pareto plot, where data0 is the standard condition
        and data1 is a perturbation. In addition, use a colormap for the data1
        scatter plot (using the label 'c').
    """
    if c0 is None:
        c0 = D.PARETO_NEUTRAL_COLOR
    if c1 is None:
        c1 = D.PARETO_STRONG_COLOR

    # a grey Pareto plot for data0
    plot_basic_pareto(data0, ax, x, y, s=s, marker=marker,
                      edgecolors=(1, 0.7, 0.7),
                      facecolors=(1, 0.7, 0.7),
                      paretofacecolors=(0.5, 0, 0),
                      paretoedgecolors=(0.5, 0, 0),
                      paretosize=20,
                      label=label0, show_efm_labels=False,
                      **kwargs)

    # a full-blown Pareto plot for data1
    plot_basic_pareto(data1, ax, x, y, s=s, marker=marker,
                      edgecolors=(0.7, 0.7, 1),
                      facecolors=(0.7, 0.7, 1),
                      paretofacecolors=(0, 0, 0.5),
                      paretoedgecolors=(0, 0, 0.5),
                      paretosize=20,
                      label=label1, show_efm_labels=False,
                      **kwargs)

    # add lines connecting the two conditions
    if draw_lines:
        data = data0[[x,y]].join(data1[[x,y]], lsuffix='', rsuffix='_1')
        for i in data.index:
            x0,y0 = data.loc[i, [x,      y     ]]
            x1,y1 = data.loc[i, [x+'_1', y+'_1']]
            ax.plot([x0, x1], [y0, y1], '-', color=(0, 0, 0), linewidth=0.2,
                    label=None, alpha=0.15)

    ax.legend(loc='upper center', fontsize=12)

def plot_scatter_with_all_labels(data, ax, x, y,
                                 facecolors='blue', edgecolors='none', alpha=1):
    """
        make plot gr vs yield for all EFMs
    """
    xdata = data.loc[:, x]
    ydata = data.loc[:, y]
    ax.scatter(xdata, ydata, s=5, marker='o', alpha=alpha,
               facecolors=facecolors, edgecolors=edgecolors)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    for i in data.index:
        ax.text(data[x][i], data[y][i], '%d' % i, fontsize=4)

def plot_sweep(data, xlabel, efm_dict, ax, legend_loc='lower right', legend_fontsize=10):
    """make line plots of gr vs parameter for all selected EFMs"""
    colors, labels = zip(*efm_dict.values())
    efm_cycler = cycler('color', colors)
    ax.set_prop_cycle(efm_cycler)
    efm_data = data.loc[efm_dict.keys(), :]
    efm_data.transpose().plot(kind='line', ax=ax, linewidth=2)

    ax.legend(labels, loc=legend_loc, fontsize=legend_fontsize, labelspacing=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'growth rate [h$^{-1}$]')
    ax.set_ylim([0, None])

def string_to_random_rgb(s, min_l=0.1, max_l=0.6, min_s=0.1, max_s=0.6):
    """
        generate 3 pseudorandom numbers from the hash-function of the name
    """
    seed = int(int(sha1(s.encode('utf-8')).hexdigest(), 16) % 1e7)
    np.random.seed(seed)
    h = rand2hue(np.random.rand())
    l = min_l + np.random.rand() * (max_l - min_l)
    s = min_s + np.random.rand() * (max_s - min_s)
    return hls_to_rgb(h, l, s)

def reaction_to_rgb(r):
    """
        Use a hash function to randomly choose a color (in a certain range
        of hues, without green, and range of saturation and brightness).
        The function is pseudorandom, but will always return the same
        color for the same input string.
    """
    if r in PREDIFINED_COLORS:
        return PREDIFINED_COLORS[r]
    else:
        return string_to_random_rgb(r, min_l=0.1, max_l=0.6, min_s=0.1, max_s=0.6)

def efm_to_hex(efm):
    """
        Use a hash function to randomly choose a color (in a certain range
        of hues, without green, and range of saturation and brightness).
        The function is pseudorandom, but will always return the same
        color for the same input string.
    """
    if efm in efm_dict:
        return efm_dict[efm]['color']
    else:
        rgb = string_to_random_rgb(str(efm), min_l=0.3, max_l=0.8, min_s=0.1, max_s=0.8)
        return matplotlib.colors.rgb2hex(rgb)

def cycle_colors(n, min_l=0.4, max_l=0.5, min_s=0.8, max_s=0.8, seed=1984):
    np.random.seed(seed)

    for x in np.linspace(0, 1, n+1)[:-1]:
        h = rand2hue(x)
        l = min_l + np.random.rand() * (max_l - min_l)
        s = min_s + np.random.rand() * (max_s - min_s)
        rgb = hls_to_rgb(h, l, s)
        yield matplotlib.colors.rgb2hex(rgb)

def allocation_area_plot(data, ax0=None, ax1=None, xlabel='',
                         n_best=10):
    """
        data - a DataFrame of index=parameter, columns=reactions, values=cost
    """
    normed_data = data.div(data.sum(axis=1), axis=0)

    # sort the reactions in decreasing order of the mean value across
    # all contidions
    sorted_reactions = normed_data.mean(axis=0).sort_values(ascending=False).index
    sig_cols = sorted_reactions[:n_best]
    rem_cols = sorted_reactions[n_best:]

    remainder = data[rem_cols].sum(1)
    remainder.name = REMAINDER_L
    lumped_data = data[sig_cols].join(remainder).abs()

    normed_remainder = normed_data[rem_cols].sum(1)
    normed_remainder.name = REMAINDER_L
    lumped_normed_data = normed_data[sig_cols].join(normed_remainder).abs()

    if ax0 is not None:
        ax0.stackplot(lumped_data.index.tolist(), lumped_data.as_matrix().T,
                      colors=map(D.reaction_to_rgb, lumped_data.columns))
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel('absolute enzyme costs\n%s' % COST_UNITS)
        ax0.set_xscale('log')
        ax0.set_xlim(lumped_data.index.min(), lumped_data.index.max())
        ax0.legend(map(D.GET_REACTION_NAME, lumped_data.columns), fontsize='small')

    if ax1 is not None:
        # Then, also make a copy with normalize values (i.e. each row
        # sums up to 1, i.e. the fraction of the total cost)
        ax1.stackplot(lumped_normed_data.index.tolist(),
                      lumped_normed_data.as_matrix().T,
                      colors=map(D.reaction_to_rgb, lumped_normed_data.columns))
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('fraction of enzyme costs')
        ax1.set_xscale('log')
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlim(lumped_normed_data.index.min(),
                     lumped_normed_data.index.max())
        if ax0 is None:
            ax1.legend(map(D.GET_REACTION_NAME, lumped_normed_data.columns), fontsize='small')

def as_base10_exp(x):
    return '$10^{%d}$' % int(np.round(np.log10(x)))
