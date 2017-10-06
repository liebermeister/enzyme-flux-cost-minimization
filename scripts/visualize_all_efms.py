# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:10:01 2015

@author: noore
"""
import re, os, zipfile, tempfile
from definitions import INPUT_SVG_FNAME, ZIP_SVG_FNAME
import pandas as pd
import definitions as D
import numpy as np
import vonda

def create_colorbar(file_out):
    colorbar_template = \
        '    <rect width="$width$" height="20" x="$x$" y="0" id="rect11" style="fill:rgb($RGB$);stroke:rgb($RGB$);"/>\n'
    
    indicatoin_template = \
        '    <path id="indication_$i$" style="fill:none;stroke:#000000;stroke-width:1.0;stroke-dasharray:none" d="m $x$,20 0,3"/>\n'
    
    tick_template = \
        '    <text x="$x$" y="30" font-family="times" font-size="8" style="fill:black">value_indication</text>\n'
    
    def CreateColorGradient(c0, c1, num):
        """
            Creates a RGB color gradient between two given colors.
        
            Arguments:
                c0    - 3-tuple with RGB values of starting color
                c1    - 3-tuple with RGB values of ending color
                num   - number of samples to generate
        """
        return [tuple(map(lambda x: int(x[0]*(1-j) + x[1]*j), zip(c0, c1)))
                for j in np.linspace(0.0, 1.0, num)]
        
    
    N_colors = 100
    fixed_points = [(256, 220, 110), (220, 100, 145), (150, 0, 150)]
    colormap = []
    for i in range(1, len(fixed_points)):
        if colormap != []:
            colormap.pop()
        colormap += CreateColorGradient(fixed_points[i-1], fixed_points[i], N_colors)
    
    colormap_size = 400
    N_ticks = 11
    
    file_out.write(
    '''<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>
<g id="colorbar" transform="translate(0,0)">''')
    
    rect_x_pos = np.linspace(0, colormap_size, len(colormap)+1)[0:-1]
    width = colormap_size / (len(colormap))
    for x, color in zip(rect_x_pos, colormap):
        line = colorbar_template.replace('$x$', '%d' % x)
        line = line.replace('$width$', '%g' % width)
        line = line.replace('$RGB$', "%d,%d,%d" % color)
        file_out.write(line)
    
    tick_x_pos = np.linspace(0.0, colormap_size, N_ticks)
    for i in range(N_ticks):
        file_out.write(indicatoin_template.replace('$i$', '%d' % i).replace('$x$', '%d' % (tick_x_pos[i])))

    for i in np.arange(N_ticks-1, -1, -1):
        file_out.write(tick_template.replace('$x$', '%d' % tick_x_pos[i]))

    file_out.write('''  </g>
  <script type="text/javascript"><![CDATA[
    var KEY = { w:87, a:65, s:83, d:68, i:73, j:74, k:75, l:76 };
    var moveSpeed = 10;
    var colorbar1 = document.getElementById("colorbar");   
    
    var xforms1 = colorbar1.transform.baseVal;  // An SVGTransformList
    var firstXForm1 = xforms1.getItem(0);     // An SVGTransform
    if (firstXForm1.type == SVGTransform.SVG_TRANSFORM_TRANSLATE){
     var firstX1 = firstXForm1.matrix.e,
         firstY1 = firstXForm1.matrix.f;
    } 

    document.documentElement.addEventListener('keydown',function(evt){
      switch (evt.keyCode){
        case KEY.w:          
          colorbar1.transform.baseVal.getItem(0).setTranslate(firstX1,firstY1-=moveSpeed);         
        break;
        case KEY.s:
          colorbar1.transform.baseVal.getItem(0).setTranslate(firstX1,firstY1+=moveSpeed);       
        break;
        case KEY.a:
          colorbar1.transform.baseVal.getItem(0).setTranslate(firstX1-=moveSpeed,firstY1);            
        break;
        case KEY.d:
          colorbar1.transform.baseVal.getItem(0).setTranslate(firstX1+=moveSpeed,firstY1);               
        break;
      }
    },false);
  ]]>
  </script>  
</svg>''')   
    file_out.flush()

def get_all_efm_rates():
    # gather the EFMs from both aerobic and anaerobic result files
    # map their numbers to the original full list values
    rates_dfs = []
    for fig_name in ['standard', 'anaerobic']:
        zip_fname = D.DATA_FILES[fig_name][0][0]
        with zipfile.ZipFile(zip_fname, 'r') as z:
            prefix, ext = os.path.splitext(os.path.basename(zip_fname))
            rates_df = pd.DataFrame.from_csv(z.open('%s/rates.csv' % prefix, 'r'),
                                             header=0, index_col=0)
            rates_dfs.append(rates_df)
    
    # drop duplicates (i.e. EFMs that can operate both aerobically and
    # anaerobically)
    rates_df = pd.concat(rates_dfs).sort_index().drop_duplicates()
    
    # we need to rename all the reaction names in rates_df to have an uppercase 'R'
    # instead of lowercase as the reaction prefix
    upper_prefix = lambda s : (s, re.sub('(^r+)', lambda match: match.group(1).upper(), s))
    rates_df.rename(columns=dict(map(upper_prefix, rates_df.columns)),
                    inplace=True)
    return rates_df

###############################################################################

rates_df = get_all_efm_rates()

with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.svg') as colormap_fp:
    create_colorbar(colormap_fp)
    
    with zipfile.ZipFile(ZIP_SVG_FNAME, 'w') as z:
    
        vmod = vonda.PVisualizer(INPUT_SVG_FNAME, reaction_suffix='R',
                                 species_suffix='',
                                 colormap=colormap_fp.name)
    
        for efm, fluxes in rates_df.iterrows():
            with tempfile.NamedTemporaryFile(delete=True, suffix='.svg') as tmpfp:
                vmod.doMapReactions(fluxes.to_dict(), scaling_mode='linear',
                                    filename_out=tmpfp.name[:-4],
                                    old_width=1.0, new_width=2.0)
                fp = z.write(tmpfp.name, arcname='efm%04d.svg' % efm)
