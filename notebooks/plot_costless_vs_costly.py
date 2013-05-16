# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint, datetime, pdb, copy
import numpy as np
import numpy.random as npr
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid, AxesGrid
from IPython.core.display import Image
# wspec moduls:
import core, storage, analytical
import visualization as viz
import utilities as utils
for mod in [core,storage,analytical,utils,viz]:     # reload the wspec modules in case the code has changed
    reload(mod)
    
#np.set_printoptions(precision=4, suppress=True, linewidth=100)
np.set_printoptions(suppress=True, linewidth=100)

# <codecell>

scenarios = [('2screen2c','costless'), ('2screen2b','costly')]
#screening_dtype = np.dtype([('pr', 'f'), ('s', 'f'), ('diff', 'f')])

data = {}
for sid,costtype in scenarios:
    rstore = storage.RunStore('/extra/flor/data/notebook_data/scenario_{0}.h5'.format(sid))
    #rstore = storage.RunStore('data/scenario_{0}.h5'.format(sid))
    scenario = rstore.get_scenario(sid)
    c = scenario['counter'][()]
    diffs = [[pr,s,d] for (pr,s,d) in scenario['screening'][:c] if s<=0.5]    # convert lenght n array of 3-tuples to array of shape (n,3)
    data[costtype] = np.array(diffs)
    rstore.close()
    print "{0:10s}:   {1} datapoints".format(costtype, len(data[costtype]))

# <codecell>

from matplotlib.colors import LinearSegmentedColormap, hex2color

rd = hex2color('#960F27')   # dark red
rl = hex2color('#FBE6DA')   # light red
bl = hex2color('#DEEBF2')   # light blue
bd = hex2color('#15508D')   # dark blue

v1 = 0.4981
v2 = 0.503

cdict = {'red':   [(0.,  rd[0],  rd[0]),
                   (v1,  rl[0],     1.),
                   (v2,     1.,  bl[0]),
                   (1.,  bd[0],  bd[0])],
         'green': [(0.,  rd[1],  rd[1]),
                   (v1,  rl[1],     1.),
                   (v2,     1.,  bl[1]),
                   (1.,  bd[1],  bd[1])],
         'blue':  [(0.,  rd[2],  rd[2]),
                   (v1,  rl[2],     1.),
                   (v2,     1.,  bl[2]),
                   (1.,  bd[2],  bd[2])]}
mycmap = LinearSegmentedColormap('MyCmap', cdict)

# plot the colormap:
a=outer(ones(10),arange(0,1,0.001))
figure(figsize=(10,0.4))
#grid(False)
axis('off')
imshow(a,aspect='auto',cmap=mycmap,origin="lower")

# <codecell>

def boundary(s):
    """
    Boundary between the two types of runaway processes based on Kirkpatrick (1982).

    s is the selection coefficient (acting at the trait locus)
    `boundary` returns the rejection probability at which the type changes:
    If r<boundary then the preference becomes fixed, and
    if r> boundary then the preferred trait becomes fixed.
    """
    return 2*s/(1+s)

# <codecell>

# set up figure environment:
fig_width_pt = 455                      # Get this from LaTeX using \showthe\textwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_height =fig_width *golden_mean              # height in inches
fig_size = [fig_width,fig_height]
print fig_size
params = {'backend': 'ps',
          'text.usetex': True,
          'text.family': 'sans-serif',
          'text.latex.preamble': [r"\usepackage{mathtools}"],   # we need this for the \mathmakebox command
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
ticks_font = font_manager.FontProperties(
    family='Helvetica',
    style='normal',
    size=8,
    weight='normal',
    stretch='normal')

fig = figure(1, figsize=fig_size)
fig.subplots_adjust(left=0.05, right=0.98)
grid = AxesGrid(fig, 131, # similar to subplot(132)
    nrows_ncols = (1, 2),
    axes_pad = 0.0,
    share_all=True,
    label_mode = "L",
    cbar_location = 'right',
    cbar_mode="single"
)

for i,costtype in enumerate(['costless', 'costly']):
    #plt.subplot(1,2,i+1)
    
    # actual data:
    X = data[costtype][:,0]
    Y = data[costtype][:,1]
    Z = data[costtype][:,2]
    
    xmin, xmax = 0., 1.
    ymin, ymax = 0., 0.5
    
    # generate griddata for contour plot:
    numspaces = 200   #3*int(math.sqrt(n))
    xi = linspace(xmin, xmax, numspaces)
    yi = linspace(ymin, ymax, numspaces)
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')
    
    im = grid[i].imshow(zi, extent=(xmin,xmax,ymin,ymax), 
        cmap=mycmap, 
        norm = matplotlib.colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False), 
        vmin=-1., vmax=1., 
        origin='lower', 
        aspect='auto', 
        interpolation='nearest')  # use default interpolation
    #plt.clim(-1., 1.)
    #plt.xlim(xmin, xmax)
    #plt.ylim(ymin, ymax)
    ax = gca()
    ax.grid(False)
    ax.set_aspect(2.)
    
    # uncomment the following line to show screening data points:
    #plt.scatter(X, Y, marker='o', color='k', alpha=0.2, s=0.5)
    
    #plot([0.18,0.18], [0., 1], 'k-', lw=1)
    #plot([0.,1.], [0.02, 0.02], 'k-', lw=1)
    #plot([0.,0.45], [0.05, 0.5], 'k-', lw=1)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax.xaxis.labelpad = 17
    ax.yaxis.labelpad = 17
    ax.set_xticklabels(ax.get_xticks(), ticks_font)
    ax.set_yticklabels(ax.get_yticks(), ticks_font)
    if costtype == 'costless':
        plt.text(-0.15, 0.52, r'\Large{A}')
        plt.xlabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Mating preference} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$',
            multialignment='left')
        plt.ylabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Viability selection} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$',
            multialignment='center')
        plt.text(0.01, 0.49, r'No spread', color='0.4', size=12, ha='left', va='top', rotation='vertical')
        plt.text(0.85, 0.2, r'\textbf{Runaway}\\(Trait fixation)', color='w', size=12, ha='right', va='bottom', multialignment='center')
        plt.text(0.12, 0.25, r'\textbf{Runaway}\\(Preference fixation)', color='0.4', size=12,ha='left', va='bottom', multialignment='center', rotation=70)
        plt.text(0., -0.025, r'Rejection probability, $\ r$', size=8, ha='left', va='top')
        plt.text(-0.085, 0., r'Selection coefficient, $\ s$', size=8, ha='left', va='bottom', rotation='vertical')
    elif costtype == 'costly':
        plt.text(-0.15, 0.52, r'\Large{B}')
        plt.xlabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Mating preference} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$',
    multialignment='left')
        plt.ylabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Viability selection} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$',
            multialignment='center')
        plt.text(0.08, 0.25, r'No spread', color='0.4', size=12, ha='center', va='center', multialignment='center', rotation='vertical')
        plt.text(0.63, 0.3, r'\textbf{Reinforcement}', color='w', size=12, ha='center', va='center')
        plt.text(0.9, 0.085, r'\textbf{Runaway}', color='w', size=12, ha='right', va='top')
        plt.text(0., -0.025, r'Rejection probability, $\ r$', size=8, ha='left', va='top')
        plt.text(-0.09, 0., r'Selection coefficient, $\ s$', size=8, ha='left', va='bottom', rotation='vertical')
# add a colorbar to the right subplot:
plt.colorbar(im, cax = grid.cbar_axes[0], ticks=[-1., -0.75, -0.5, -0.25, -0.01, 0.01, 0.25, 0.5, 0.75, 1.])
grid.cbar_axes[0].colorbar(im)

for cax in grid.cbar_axes:
    cax.toggle_label(False)
        
# This affects all axes as share_all = True.
#grid.axes_llc.set_xticks([-2, 0, 2])
#grid.axes_llc.set_yticks([-2, 0, 2])

grid.cbar_axes[0].set_yticklabels(['--1.0', '--0.75', '--0.5', '--0.25', '0.0', '', '0.25', '0.5', '0.75', '1.0'])
grid.cbar_axes[0].set_ylabel(r'$\xleftarrow{\mathmakebox[8em]{\textstyle\text{decreasing}}}$ {\large Divergence} $\xrightarrow{\mathmakebox[8em]{\textstyle\text{increasing}}}$')
plt.setp(grid.cbar_axes[0].yaxis.get_ticklines(minor=False), markersize=0)

plt.savefig('images/costless_vs_costly.pdf')  #, bbox_inches='tight', dpi=600)

# <codecell>

# set up figure environment:
fig_width_pt = 455                           # Get this from LaTeX using \showthe\textwidth
inches_per_pt = 1.0/72.27                    # Convert pt to inches
fig_width  = fig_width_pt * inches_per_pt   # width in inches
fig_height = fig_width * 0.5                 # height in inches
fig_size   = [fig_width,fig_height]
print fig_size
params = {'backend': 'ps',
          'text.usetex': True,
          'text.family': 'sans-serif',
          'text.latex.preamble': [r"\usepackage{mathtools}"],   # we need this for the \mathmakebox command
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
ticks_font = font_manager.FontProperties(
    family='Helvetica',
    style='normal',
    size=10,
    weight='normal',
    stretch='normal')

fig = figure(1, figsize=fig_size)
fig.subplots_adjust(left=0.0, right=0.98)
grid = AxesGrid(fig, 111, # similar to subplot(132)
    nrows_ncols = (1, 2),
    axes_pad = 0.7,
    share_all = False,
    label_mode = 'all',
    cbar_location = 'right',
    cbar_mode = 'single',
    cbar_pad = 0.2
)

for i,costtype in enumerate(['costless', 'costly']):    
    # actual data:
    X = data[costtype][:,0]
    Y = data[costtype][:,1]
    Z = data[costtype][:,2]
    
    xmin, xmax = 0., 1.
    ymin, ymax = 0., 0.5
    
    # generate griddata for contour plot:
    numspaces = 200   #3*int(math.sqrt(n))
    xi = linspace(xmin, xmax, numspaces)
    yi = linspace(ymin, ymax, numspaces)
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')
    norm = matplotlib.colors.normalize(vmin = -1.0, vmax = 1.0, clip = True)
    
    ax = grid[i]
    im = ax.imshow(zi,
        extent = [xmin,xmax,ymin,ymax], 
        cmap = mycmap, 
        norm = norm, 
        vmin = -1.,
        vmax = 1., 
        origin = 'lower', 
        aspect = 'auto', 
        interpolation = 'nearest')  # use default interpolation
    
    # uncomment the following line to show screening data points:
    #plt.scatter(X, Y, marker='o', color='k', alpha=0.2, s=0.5)
    
    #plot([0.18,0.18], [0., 1], 'k-', lw=1)
    #plot([0.,1.], [0.02, 0.02], 'k-', lw=1)
    #plot([0.,0.45], [0.05, 0.5], 'k-', lw=1)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.labelpad = 17
    ax.yaxis.labelpad = 17
    ax.set_xticklabels(ax.get_xticks(), ticks_font)
    ax.set_yticklabels(ax.get_yticks(), ticks_font)
    ax.set_aspect(0.5)
    
    ax.grid(False)
    if costtype == 'costless':
        ax.text(-0.15, 0.52, r'\Large{A}')
        ax.set_xlabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Mating preference} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$', multialignment='left')
        ax.set_ylabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Viability selection} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$', multialignment='center')
        ax.text(0.01, 0.49, r'No spread of mating preference', color='0.4', size=12, ha='left', va='top', rotation='vertical')
        ax.text(0.85, 0.2, r'\textbf{Runaway}\\(Trait fixation)', color='w', size=12, ha='right', va='bottom', multialignment='center')
        ax.text(0.12, 0.25, r'\textbf{Runaway}\\(Preference fixation)', color='0.4', size=12,ha='left', va='bottom', multialignment='center', rotation=70)
        ax.text(0., -0.025, r'Rejection probability, $\ r$', size=10, ha='left', va='top')
        ax.text(-0.085, 0., r'Selection coefficient, $\ s$', size=10, ha='left', va='bottom', rotation='vertical')
    elif costtype == 'costly':
        ax.text(-0.15, 0.52, r'\Large{B}')
        ax.set_xlabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Mating preference} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$', multialignment='left')
        ax.set_ylabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Viability selection} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$', multialignment='center')
        ax.text(0.08, 0.25, r'No spread of mating preference', color='0.4', size=12, ha='center', va='center', multialignment='center', rotation='vertical')
        ax.text(0.63, 0.3, r'\textbf{Reinforcement}', color='w', size=12, ha='center', va='center')
        ax.text(0.9, 0.085, r'\textbf{Runaway}', color='w', size=12, ha='right', va='top')
        ax.text(0., -0.025, r'Rejection probability, $\ r$', size=10, ha='left', va='top')
        ax.text(-0.09, 0., r'Selection coefficient, $\ s$', size=10, ha='left', va='bottom', rotation='vertical')
    
# add a colorbar:
cbar = plt.colorbar(im, cax=grid.cbar_axes[0], ticks=[-1., -0.75, -0.5, -0.25, -0.01, 0.01, 0.25, 0.5, 0.75, 1.])
#cbar.ax.set_aspect(0.1)

# This affects all axes as share_all = True.
#grid.axes_llc.set_xticks([0, 1])
#grid.axes_llc.set_yticks([0, 0.5])

cbar.ax.set_yticklabels(['--1.0', '--0.75', '--0.5', '--0.25', '0.0', '', '0.25', '0.5', '0.75', '1.0'])
cbar.ax.set_ylabel(r'$\xleftarrow{\mathmakebox[8em]{\textstyle\text{decreasing}}}$ {\large Divergence} $\xrightarrow{\mathmakebox[8em]{\textstyle\text{increasing}}}$')
plt.setp(cbar.ax.yaxis.get_ticklines(minor=False), markersize=0)

plt.savefig('images/costless_vs_costly.pdf')  #, bbox_inches='tight', dpi=600)

# <codecell>

Grid??

# <codecell>

pwd

# <codecell>

import numpy as np
import numpy.random as npr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

fig = figure(1, figsize=[12,10])
fig.subplots_adjust(left=0.05, right=0.98)

grid = AxesGrid(fig, 111,
    nrows_ncols = (1, 2),
    axes_pad = 0.2,
    share_all = False,
    label_mode = 'L',
    cbar_location = 'right',
    cbar_mode = 'single',
    cbar_pad = 0.2
)

for i in range(2):  
    xmin, xmax = 0., 1.
    ymin, ymax = 0., 0.5
    zmin, zmax = -1., 1.
    
    # create random data:
    N = 100
    X = xmin + (xmax-xmin)*npr.random((N,))          # x_i in [0, 1]
    Y = ymin + (ymax-ymin)*npr.random((N,))          # y_i in [0, 0.5]
    Z = zmin + (zmax-zmin)*npr.random((N,))          # z_i in [-1, 1]
    
    # generate griddata for contour plot:
    numspaces = np.sqrt(N)
    xi = linspace(xmin, xmax, numspaces)
    yi = linspace(ymin, ymax, numspaces)
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')
    norm = matplotlib.colors.normalize(vmin=zmin, vmax=zmax)
    
    ax = grid[i]
    im = ax.imshow(zi,
        extent = [xmin,xmax,ymin,ymax],  
        norm = norm, 
        vmin = zmin,
        vmax = zmax, 
        origin = 'lower', 
        aspect = 'auto', 
        interpolation = 'nearest')
    
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)
    #ax.xaxis.labelpad = 17
    #ax.yaxis.labelpad = 17
    #ax.set_xticklabels(ax.get_xticks(), ticks_font)
    #ax.set_yticklabels(ax.get_yticks(), ticks_font)
    #ax.set_aspect(0.5)
    
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
# add a colorbar:
cbar = plt.colorbar(im, cax=grid.cbar_axes[0])
cbar.ax.set_ylabel('color level')

# <codecell>


