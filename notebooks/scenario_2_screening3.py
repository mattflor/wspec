# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Scenario 2

# <codecell>

import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint, datetime, pdb, copy
import numpy as np
import numpy.random as npr
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
from IPython.core.display import Image
# wspec moduls:
import core, storage, analytical
import visualization as viz
import utilities as utils
for mod in [core,storage,analytical,utils,viz]:     # reload the wspec modules in case the code has changed
    reload(mod)
    
#np.set_printoptions(precision=4, suppress=True, linewidth=100)
np.set_printoptions(suppress=True, linewidth=100)
npr.seed(230985)

# <markdowncell>

# ## 1. Scenario features
# 
# * Two populations linked by migration
# 
# * Initial state:
# 
#     * Trait T1 adaptive and fixed in population 1, T2 in population 2
# 
#     * Preference allele P0 (non-discriminating) fixed in both populations
# 
# * Population 1 is uninfected, *Wolbachia* infection in population 2
# 
# * Order of events:
# 
#     1.  Secondary contact  
#         $\rightarrow$  selection-migration equilibrium
# 
#     2.  Introduction of a preference for T1 in population 1, P1 (T1), and of a preference for T2 in population 2, P2 (T2)  
#         $\rightarrow$  new equilibrium
# 
# <img src="https://docs.google.com/drawings/d/1PI65vCcdE8Yv5Sa5cS_NrZ3QZzjU8nMVnLLMJycal04/pub?w=691&h=577">

# <markdowncell>

# ## 2. Configuration
# 
# To configure the simulation scenario, we need to specify gene loci, alleles, and parameters.
# 
# ### 2.1 Loci and alleles
# 
# Populations are treated as loci, as would be the cytotype. In numpy terms, each locus is represented by an array axis.
# 
# We keep separate lists rather than a dictionary because we need to preserve the locus and allele orders.

# <codecell>

LOCI = ['population', 'trait', 'preference', 'cytotype']
ALLELES = [['pop1', 'pop2'],
           ['T1', 'T2'],
           ['P0', 'P1', 'P2'],
           ['U', 'W']
          ]
print utils.loci2string(LOCI, ALLELES)

# <codecell>

def popdiff(metapop):
    """
    Measure frequency differences between populations. This function returns a NxN matrix containing 
    values between 0 and L-1 where N is the number of populations and L is the number of loci.
    """
    n = metapop.n_pops
    d = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i,n):
            for loc in metapop.loci[1:]:
                sums = metapop.get_sums(loc)
                d[i,j] += np.max( np.abs(sums[i] - sums[j]) )
                d[j,i] = d[i,j]
    return d

# <headingcell level=3>

# 2.2 Parameters

# <markdowncell>

# Beware of the `step` parameter as the size of the HDF5 database is strongly affected by its value. If we are not interested in every single dynamical process but rather in the final outcome then might set `step = 1e6` which will only record special states like initial frequencies, equilibria, or when new alleles have been introduced.

# <codecell>

sid = '2screen3'     # scenario id
PARAMETERS = {
    'lCI': (0.9, 'CI level'),                   # level of cytoplasmic incompatibility
    't': (0.9, 'transmission rate'),            # transmission of Wolbachia
    'f': (0.1, 'fecundity reduction'),          # Wolbachia-infected females are less fecund
    'm': (0.01, 'migration rate'),              # symmetric migration
    'intro': (0.001, 'introduction frequency'), # introduction frequency of preference mutant allele
    'eq': (1e-6, 'equilibrium threshold'),      # equilibrium threshold (total frequency change)
    'nmin': (1000, 'min generation'),            # run at least `nmin` generations in search of equilibrium
    'nmax': (300000, 'max generation'),         # max number of generations to iterate for each stage of the simulation
    'step': (10, 'storage stepsize')            # store metapopulation state every `step` generations
}
# For mating preference parameters, we use a different notation:
pr = 0.8
trait_preferences = {                        # female mating preferences (rejection probabilities)
    'P0': {'baseline': 0.},
    'P1': {'baseline': pr, 'T1': 0.},
    'P2': {'baseline': pr, 'T2': 0.}
}
PARAMETERS = utils.add_preferences(PARAMETERS, trait_preferences)
# make parameter names locally available:
config = utils.configure_locals(LOCI, ALLELES, PARAMETERS)
locals().update(config)

print 'Parameters that are the same for all runs in the screening:\n'
print utils.params2string(PARAMETERS)

# <markdowncell>

# Critical migration rate for these *Wolbachia* parameters:

# <codecell>

print 'm_crit =', np.min(analytical.mcrit_IMA(f, lCI, s, t), analytical.mcrit_UMA(f, lCI, s, t))

# <markdowncell>

# We'll generate random transition and rejection probabilities, run the simulation, and check the outcome via the `popdiff` function the results of which are stored as well.

# <codecell>

n = 4000

screening_dtype = np.dtype([('pt', 'f'), ('s', 'f'), ('diff', 'f')])
rstore = storage.RunStore('/extra/flor/data/notebook_data/scenario_{0}.h5'.format(sid))
# select existing scenario, initialize a new one if this fails:
try:
    scenario = rstore.select_scenario(sid, verbose=False)
    count = scenario['counter'][()]
    maxcount = len(scenario['screening'])
except:
    scenario = rstore.create_scenario(sid, labels=(LOCI,ALLELES))
    # scenario = rstore['current']['scenario']
    scenario.create_dataset('counter', (), 'i')    # integer counter for the screening runs; initialized at 0
    count = 0
    maxcount = max(n, 10)
    scenario.create_dataset('screening', (maxcount,), screening_dtype, maxshape=(None,))
    
if n >= maxcount:
    scenario['screening'].resize((n,))

# <markdowncell>

# The actual screening:

# <codecell>

for i in range(n):
    # we use a U-shaped beta function (low and high values more likely than intermediate ones)
    # to better explore the corners:
    if i <= 2000:
        pt = npr.beta(a=0.7, b=0.7)
        s  = 0.5*npr.beta(a=0.7, b=0.7)
    elif (2000 < i < 2500) or (3500 < i <= 4000):
        s  = 0.5*npr.random()
        pt_min = (s + 4.)/5.
        pt = pt_min + (1.-pt_min)*npr.random()
    elif 2500 <= i <= 3500:
        s  = 0.5*npr.random()
        pt_min = 0.85 - np.sqrt(s/1.6)
        pt = pt_min + np.min(0.2, 1.-pt_min)*npr.random()
        
    PARAMETERS['s'] = (s, 'selection coefficient')        # selection advantage for adaptive trait
    PARAMETERS['pt'] = (pt, 'transition probability')       # probability of transition into another mating round
    
    rid = 'pt{0:.12f}_s{1:.12f}'.format(pt,s)      # generate id of simulation run based on pr and pt values
    
    # make parameter names locally available again:
    config = utils.configure_locals(LOCI, ALLELES, PARAMETERS)
    locals().update(config)

    max_figwidth = 15
    figheight = 5
    w = min( N_POPS*(N_LOCI-1), max_figwidth )    # figure width: npops*(nloci-1) but at most 15
    figsize = [w, figheight]
    show_progressbar = False          # BEWARE: enabling progressbar slows down the simulation significantly!
    
    overwrite_run = False
    data_available = False
    
    # select existing run, initialize a new one if this fails:
    try:   
        rstore.select_run(rid, verbose=False)
        data_available = True
        special_states = list( rstore.get_special_states()[::-1] )
    except:
        rstore.init_run(rid, PARAMETERS, FSHAPE, init_len=100)
    # check whether parameters are identical:
    pars = rstore.get_parameters()
    if not utils.parameters_equal(pars, PARAMETERS, verbose=False):
        data_available = False
        if not overwrite_run:
            raise ValueError('parameter values differ from stored values; set `overwrite_run` to `True` in order to overwrite run')
        else:
            # print 'overwriting run...'
            rstore.remove_run(rid, sid)
            rstore.init_run(rid, PARAMETERS, FSHAPE, init_len=100)
    
    if data_available:
        pass
    else:
        weights = {
            'migration': None,
            'viability_selection': None,
            'constant_reproduction': None,
            'dynamic_reproduction': []
        }
        mig = np.array(
            [[1-m,      m ],
             [  m,    1-m ]], float)
        M = core.MigrationWeight(
            name='migration',
            axes=['target', 'source'],
            config=config,
            arr=mig,
            m=m
        )
        weights['migration'] = M.extended()
        # print M
        
        vsarr = np.array(
            [[ 1+s,  1  ],
             [   1,  1+s]], float
        )
        VS = core.ViabilityWeight(
            name='viability selection',
            axes=['population','trait'],
            config=config,
            arr=vsarr,
            s=s
        )
        weights['viability_selection'] = VS.extended()
        # print VS
        
        TP = core.GeneralizedPreferenceWeight(
            name='trait preference',
            axes=['population', 'female_preference', 'male_trait'],
            pref_desc = trait_preferences,
            config=config,
            unstack_levels=[2],
            pt=pt
        )
        weights['dynamic_reproduction'].append( (TP, ['trait']) )
        # print TP
        
        CI = core.ReproductionWeight(
            name='cytoplasmic incompatibility',
            axes=['male_cytotype', 'offspring_cytotype'],
            config=config,
            unstack_levels=[1],
            lCI=lCI
        )
        CI.set( np.array([[1, 1], [1-lCI, 1]], float ) )
        CI_ = CI.extended()
        # print CI
    
        T = core.ReproductionWeight(
            name='cytotype inheritance',
            axes=['female_cytotype', 'offspring_cytotype'],
            config=config,
            unstack_levels=[1],
            t=t
        )
        T.set( np.array( [[1, 0], [1-t, t]], float ) )
        T_ = T.extended()
        # print T
        
        F = core.ReproductionWeight(
            name='fecundity reduction',
            axes=['female_cytotype'],
            config=config,
            f=f
        )
        F.set( np.array([1, 1-f], float) )
        F_ = F.extended()
        # print F
        
        IP = core.ReproductionWeight(
            name='preference inheritance',
            axes=['female_preference', 'male_preference', 'offspring_preference'],
            config=config,
            unstack_levels=[2]
        )
        n_alleles = len(ALLELES[LOCI.index('preference')])
        IP.set( utils.nuclear_inheritance(n_alleles) )
        IP_ = IP.extended()
        # print IP
        
        IT = core.ReproductionWeight(
            name='trait inheritance',
            axes=['female_trait', 'male_trait', 'offspring_trait'],
            config=config,
            unstack_levels=[2]
        )
        n_alleles = len(ALLELES[LOCI.index('trait')])
        IT.set( utils.nuclear_inheritance(n_alleles) )
        IT_ = IT.extended()
        # print IT
        
        R_ = CI_ * T_ * F_ * IP_ * IT_
        weights['constant_reproduction'] = R_
        
        # INITIAL STATE:
        starttime = time.time()                  # take time for timing report after simulation run
        startfreqs = np.zeros(FSHAPE)
        startfreqs[0,0,0,0] = 1.                   # pop1-T1-P0-U
        startfreqs[1,1,0,1] = 1.                   # pop2-T2-P0-W
        metapop = core.MetaPopulation(
            startfreqs,
            config=config,
            generation=0,
            name='metapopulation'
        )
        rstore.record_special_state(metapop.generation, 'start')
        rstore.dump_data(metapop)
        
        # MIGRATION-SELECTION EQUILIBRIUM:
        metapop.run(
            nmax,
            weights,
            thresh_total=eq,
            n_min=nmin,
            step=step,
            runstore=rstore,
            progress_bar=show_progressbar,
            verbose=False
        )
        # make a copy of the metapopulation for later calculation of frequency differences:
        #initial = copy.copy(metapop)
        # frequency difference score before introduction of preference alleles:
        base_diff = popdiff(metapop)[0,1]
        
        # INTRODUCTION OF PREFERENCE MUTANTS:
        intro_allele = 'P1'
        metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=False)
        rstore.dump_data(metapop)
        rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
        
        intro_allele = 'P2'
        metapop.introduce_allele('pop2', intro_allele, intro_freq=intro, advance_generation_count=True)
        rstore.dump_data(metapop)
        rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
        
        # FINAL EQUILIBRIUM:
        metapop.run(
            nmax,
            weights,
            thresh_total=eq,
            n_min=nmin,
            step=step,
            runstore=rstore,
            progress_bar=show_progressbar,
            verbose=False
        )
        
        #diff = popdiff2(metapop, initial)[0,1]
        diff = popdiff(metapop)[0,1] - base_diff
        scenario['screening'][i] = (pt, s, diff)
        scenario['counter'][()] += 1
        
        print '\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        print 'i = {0}\npt = {1}\ns = {2}\ndiff = {3}\n'.format(i,pt,s,diff)
        print metapop
        print metapop.overview()
        print
        print utils.timing_report(starttime, metapop.generation)
        rstore.flush()
        
        # fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)
        # fig = rstore.plot_sums(figsize=[max_figwidth, figheight])

# <markdowncell>

# Collecting and printing outcomes for all parameter combinations:

# <codecell>

c = scenario['counter'][()]
diffs = np.array( [[x,y,z] for (x,y,z) in scenario['screening'][:c]] )    # convert lenght n array of 3-tuples to array of shape (n,3)
print '   pt      s       diff   '
print '--------------------------'
print diffs

# <headingcell level=2>

# Plot

# <markdowncell>

# We use a custom colormap which changes from dark to light red, then from light to dark blue, and which has a small segment of white where red and blue touch each other.

# <codecell>

from matplotlib.colors import LinearSegmentedColormap, hex2color

rd = hex2color('#960F27')   # dark red
rl = hex2color('#FBE6DA')   # light red
bl = hex2color('#DEEBF2')   # light blue
bd = hex2color('#15508D')   # dark blue

w = 0.005  # width of the white segment
v1 = 0.5 - w/2.
v2 = 0.5 + w/2.

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
grid(False)
axis('off')
imshow(a,aspect='auto',cmap=mycmap,origin="lower")

# <codecell>

# set up figure environment:
fig_width_pt = 500.0                    # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width *0.75             # height in inches
fig_size = [fig_width,fig_height]
params = {'backend': 'ps',
          'text.usetex': True,
          'text.family': 'sans-serif',
          'text.latex.preamble': [r"\usepackage{mathtools}"],
          'axes.labelsize': 10,
          'text.fontsize': 12,
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

# actual data:
X = diffs[:,0]
Y = diffs[:,1]
Z = diffs[:,2]

xmin, xmax = 0., 1.
ymin, ymax = 0., 0.5

# generate griddata for contour plot:
numspaces = 2*int(math.sqrt(n))
xi = linspace(xmin, xmax, numspaces)
yi = linspace(ymin, ymax, numspaces)
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')

figure(1, figsize=fig_size)
plt.imshow(zi, extent=(xmin,xmax,ymin,ymax), 
    cmap=mycmap, 
    norm = matplotlib.colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False), 
    vmin=-1., vmax=1., 
    origin='lower', 
    aspect='auto', 
    interpolation=None)  # use default interpolation
plt.clim(-1., 1.)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
ax = gca()
ax.grid(False)
ax.set_aspect(2.)

# add a colorbar:
cbar = plt.colorbar(ticks=[-1., -0.75, -0.5, -0.25, -0.01, 0.01, 0.25, 0.5, 0.75, 1.])
cax = cbar.ax
cax.set_yticklabels(['--1.0', '--0.75', '--0.5', '--0.25', '0.0', '', '0.25', '0.5', '0.75', '1.0'])
cax.set_ylabel(r'$\xleftarrow{\mathmakebox[8em]{\textstyle\text{decreasing}}}$ {\large Divergence} $\xrightarrow{\mathmakebox[8em]{\textstyle\text{increasing}}}$')
plt.setp(cax.yaxis.get_ticklines(minor=False), markersize=0)

# uncomment the following line to show screening data points:
#plt.scatter(X, Y, marker='o', color='k', alpha=0.2, s=0.5)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
ax.xaxis.labelpad = 17
ax.yaxis.labelpad = 17
ax.set_xticklabels(ax.get_xticks(), ticks_font)
ax.set_yticklabels(ax.get_yticks(), ticks_font)
plt.xlabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{high}}}$ {\large Preference costs} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{low}}}$',
    multialignment='left')
plt.ylabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Viability selection} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$',
    multialignment='center')
plt.text(0.1, 0.15, r'No spread of\\mating preferences', color='0.4', size=12, ha='left', multialignment='center')
plt.text(0.55, 0.4, r'\textbf{Reinforcement}', color='w', size=14, ha='left')
plt.text(0.99, 0.02, r'\textbf{Runaway}', color='w', size=14, ha='right', va='bottom', rotation='vertical')
plt.text(0., -0.025, r'Transition probability, $\ p$', ha='left', va='top')
plt.text(-0.1, 0., r'Selection coefficient, $\ s$', ha='left', va='bottom', rotation='vertical')
#plt.savefig('/home/flor/Documents/Manuscripts/Runaways and reinforcement/Screening_costs_vs_preferencestrength.pdf')
plt.savefig('images/costly_s_vs_p.pdf')

# <codecell>

len(diffs)

# <codecell>

rstore.close()

# <codecell>

# set up figure environment:
fig_width_pt = 500.0                    # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width *0.75             # height in inches
fig_size = [fig_width,fig_height]
params = {'backend': 'ps',
          'text.usetex': True,
          'text.family': 'sans-serif',
          'text.latex.preamble': [r"\usepackage{mathtools}"],
          'axes.labelsize': 10,
          'text.fontsize': 12,
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

# actual data:
X = diffs[:,0]
Y = diffs[:,1]
Z = diffs[:,2]

xmin, xmax = 0.5, 1.
ymin, ymax = 0., 0.5

# generate griddata for contour plot:
numspaces = 2*int(math.sqrt(n))
xi = linspace(xmin, xmax, numspaces)
yi = linspace(ymin, ymax, numspaces)
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')

figure(1, figsize=fig_size)
plt.imshow(zi, extent=(xmin,xmax,ymin,ymax), 
    cmap=mycmap, 
    norm = matplotlib.colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False), 
    vmin=-1., vmax=1., 
    origin='lower', 
    aspect='auto', 
    interpolation=None)  # use default interpolation
plt.clim(-1., 1.)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
ax = gca()
ax.grid(False)
ax.set_aspect(1.)

# add a colorbar:
cbar = plt.colorbar(ticks=[-1., -0.75, -0.5, -0.25, -0.01, 0.01, 0.25, 0.5, 0.75, 1.])
cax = cbar.ax
cax.set_yticklabels(['--1.0', '--0.75', '--0.5', '--0.25', '0.0', '', '0.25', '0.5', '0.75', '1.0'])
cax.set_ylabel(r'$\xleftarrow{\mathmakebox[8em]{\textstyle\text{decreasing}}}$ {\large Divergence} $\xrightarrow{\mathmakebox[8em]{\textstyle\text{increasing}}}$')
plt.setp(cax.yaxis.get_ticklines(minor=False), markersize=0)

# uncomment the following line to show screening data points:
#plt.scatter(X, Y, marker='o', color='k', alpha=0.2, s=0.5)

plt.xlim(xmin, xmax)
ax.set_xlim(ax.get_xlim()[::-1])
plt.ylim(ymin, ymax)
ax.xaxis.labelpad = 17
ax.yaxis.labelpad = 17
ax.set_xticklabels(ax.get_xticks(), ticks_font)
ax.set_yticklabels(ax.get_yticks(), ticks_font)
plt.xlabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{small}}}$ {\large Preference costs} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{large}}}$',
    multialignment='left')
plt.ylabel(r'$\xleftarrow{\mathmakebox[6em]{\textstyle\text{weak}}}$ {\large Viability selection} $\xrightarrow{\mathmakebox[6em]{\textstyle\text{strong}}}$',
    multialignment='center')
plt.text(0.52, 0.03, r'No spread of\\mating preferences', color='0.4', size=12, ha='right', multialignment='center')
plt.text(0.85, 0.35, r'\textbf{Reinforcement}', color='w', size=14, ha='left')
plt.text(0.98, 0.03, r'\textbf{Runaway}', color='w', size=14, ha='left', va='bottom', rotation='vertical')
plt.text(1., -0.025, r'Transition probability, $\ p$', ha='left', va='top')
plt.text(1.05, 0., r'Selection coefficient, $\ s$', ha='left', va='bottom', rotation='vertical')
#plt.savefig('/home/flor/Documents/Manuscripts/Runaways and reinforcement/Screening_costs_vs_preferencestrength.pdf')
plt.savefig('images/costly_s_vs_p.pdf')

# <codecell>


