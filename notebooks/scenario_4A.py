# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

#     Scenario 4

# <codecell>

import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint, datetime, pdb
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import Image
# wspec moduls:
import core, storage, analytical
import visualization as viz
import utilities as utils
for mod in [core,storage,analytical,utils,viz]:     # reload the wspec modules in case the code has changed
    reload(mod)
    
np.set_printoptions(precision=4, suppress=True, linewidth=100)

# <markdowncell>

# ## 1. Scenario features
# 
# * Two populations linked by migration
# 
# * Initial state:
#     * Trait divergence: T1 adaptive and fixed in population 1, T2 in population 2
#     * Divergence at the preference locus: P1 (T1) fixed in population 1, P2 (T2) in population 2
#     * *Wolbachia* infection in population 2, Population 1 is uninfected
# 
# * Order of events:
#     1. Secondary contact  $\rightarrow$  selection-migration equilibrium
#     2. Introduction of a non-discriminating preference mutant as well as stronger mating preferences in both populations: P3 (T1), P4 (T2), and P0 (--)  $\rightarrow$  new equilibrium
# 
# <img src="https://docs.google.com/drawings/d/1Zl6QN4SsaQoXSgKumxyx-f52-2ccVntiu7jxbkxO1oM/pub?w=691&amp;h=577">

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
           ['P1', 'P2', 'P3', 'P4', 'P5'],
           ['U', 'W']
          ]
print utils.loci2string(LOCI, ALLELES)

# <headingcell level=3>

# 2.2 Parameters

# <markdowncell>

# Scenario and simulation run id's, and parameters:

# <codecell>

sid = 4     # scenario id
rid = 'A'     # id of simulation run

# <codecell>

PARAMETERS = {
    'lCI': (0.9, 'CI level'),                     # level of cytoplasmic incompatibility
    't': (0.9, 'transmission rate'),             # transmission of Wolbachia
    'f': (0.1, 'fecundity reduction'),            # Wolbachia-infected females are less fecund
    'm': (0.01, 'migration rate'),                # symmetric migration
    's': (0.1, 'selection coefficient'),           # selection advantage for adaptive trait
    'pt': (0.9, 'transition probability'),       # probability of transition into another mating round
    'intro': (0.001, 'introduction frequency'),   # introduction frequency of preference mutant allele
    'eq': (1e-6, 'equilibrium threshold'),      # equilibrium threshold (total frequency change)
    'nmax': (30000, 'max generation'),          # max number of generations to iterate for each stage of the simulation
    'step': (10, 'storage stepsize')            # store metapopulation state every `step` generations
}
# For mating preference parameters, we use a different notation:
trait_preferences = {                        # female mating preferences (rejection probabilities)
    'P1': {'baseline': 0.4, 'T1': 0.},
    'P2': {'baseline': 0.4, 'T2': 0.},
    'P3': {'baseline': 1., 'T1': 0.},
    'P4': {'baseline': 1., 'T2': 0.},
    'P5': {'baseline': 0.}
}
PARAMETERS = utils.add_preferences(PARAMETERS, trait_preferences)
# make parameter names locally available:
config = utils.configure_locals(LOCI, ALLELES, PARAMETERS)
locals().update(config)
# print all parameters:
print utils.params2string(PARAMETERS)

# <markdowncell>

# Simulation run data is stored in an HDF5 file (`storage.RunStore` basically is a wrapper around an `h5py.File` object):

# <codecell>

overwrite_run = True
data_available = False
rstore = storage.RunStore('data/scenario_{0}.h5'.format(sid))
# select existing scenario, initialize a new one if this fails:
try:
    rstore.select_scenario(sid, verbose=False)
except:
    rstore.create_scenario(sid, labels=(LOCI,ALLELES))
# select existing run, initialize a new one if this fails:
try:   
    rstore.select_run(rid)
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
        print 'overwriting run...'
        rstore.remove_run(rid, sid)
        rstore.init_run(rid, PARAMETERS, FSHAPE, init_len=100)

# <markdowncell>

# Configure plotting:

# <codecell>

max_figwidth = 15
figheight = 5
w = min( N_POPS*(N_LOCI-1), max_figwidth )    # figure width: npops*(nloci-1) but at most 15
figsize = [w, figheight]
show_progressbar = False          # BEWARE: enabling progressbar slows down the simulation significantly!

# <headingcell level=2>

# 3. Weights

# <markdowncell>

# All weights for the different stages of the simulation are stored in a dictionary.

# <codecell>

weights = {
    'migration': None,
    'viability_selection': None,
    'constant_reproduction': None,
    'dynamic_reproduction': []
}

# <markdowncell>

# We now define all the weights we use in the simulation.
# These are in principal `ndarrays` that can be automatically extended to the appropriate dimensions by insertion of `np.newaxis` at the required positions.
# The extended weights are denoted by a trailing underscore.
# For printing, `panda.Series` are used.

# <headingcell level=3>

# 3.1 Migration

# <codecell>

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
print M

# <headingcell level=3>

# 3.2 Viability selection

# <codecell>

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
print VS

# <headingcell level=3>

# 3.3 Sexual selection (female mating preference)

# <markdowncell>

# These weights are frequency-dependent. Their final states can be found in section `Dynamic weights (final states)`.

# <headingcell level=4>

# Trait preference

# <codecell>

TP = core.GeneralizedPreferenceWeight(
    name='trait preference',
    axes=['population', 'female_preference', 'male_trait'],
    pref_desc = trait_preferences,
    config=config,
    unstack_levels=[2],
    pt=pt
)
weights['dynamic_reproduction'].append( (TP, ['trait']) )
print TP

# <headingcell level=3>

# 3.4 Reproduction

# <headingcell level=4>

# Cytoplasmic incompatibility

# <codecell>

CI = core.ReproductionWeight(
    name='cytoplasmic incompatibility',
    axes=['male_cytotype', 'offspring_cytotype'],
    config=config,
    unstack_levels=[1],
    lCI=lCI
)
CI.set( np.array([[1, 1], [1-lCI, 1]], float ) )
CI_ = CI.extended()
print CI

# <headingcell level=4>

# Cytotype inheritance (Wolbachia transmission)

# <codecell>

T = core.ReproductionWeight(
    name='cytotype inheritance',
    axes=['female_cytotype', 'offspring_cytotype'],
    config=config,
    unstack_levels=[1],
    t=t
)
T.set( np.array( [[1, 0], [1-t, t]], float ) )
T_ = T.extended()
print T

# <headingcell level=4>

# Female fecundity

# <codecell>

F = core.ReproductionWeight(
    name='fecundity reduction',
    axes=['female_cytotype'],
    config=config,
    f=f
)
F.set( np.array([1, 1-f], float) )
F_ = F.extended()
print F

# <headingcell level=4>

# Nuclear inheritance

# <markdowncell>

# Nuclear inheritance weights for all loci.

# <codecell>

IP = core.ReproductionWeight(
    name='preference inheritance',
    axes=['female_preference', 'male_preference', 'offspring_preference'],
    config=config,
    unstack_levels=[2]
)
n_alleles = len(ALLELES[LOCI.index('preference')])
IP.set( utils.nuclear_inheritance(n_alleles) )
IP_ = IP.extended()
print IP

# <codecell>

IT = core.ReproductionWeight(
    name='trait inheritance',
    axes=['female_trait', 'male_trait', 'offspring_trait'],
    config=config,
    unstack_levels=[2]
)
n_alleles = len(ALLELES[LOCI.index('trait')])
IT.set( utils.nuclear_inheritance(n_alleles) )
IT_ = IT.extended()
print IT

# <markdowncell>

# We can combine all reproduction weights that are not frequency-dependent:

# <codecell>

R_ = CI_ * T_ * F_ * IP_ * IT_
weights['constant_reproduction'] = R_

# <headingcell level=2>

# 4. Simulation

# <codecell>

m1 = analytical.mcrit_UM(f, lCI, t)     # uninfected mainland --> infected island
m2 = analytical.mcrit_IM(f, lCI, t)     # infected mainland --> uninfected island
mcrit = min(m1,m2)
print 'm_crit = %.5f' % mcrit
print 'm      =', m
assert m < mcrit
print 'p_t    =', pt
print 's      =', s
print 'p_r    =', pr_p1_baseline

# <headingcell level=3>

# 4.1  Initial state

# <codecell>

if not data_available:
    starttime = time.time()                  # take time for timing report after simulation run
    startfreqs = np.zeros(FSHAPE)
    startfreqs[0,0,0,0] = 1.                   # pop1-T1-P1-U
    startfreqs[1,1,1,1] = 1.                   # pop2-T2-P2-W
    # initialize metapopulation with start frequencies:
    metapop = core.MetaPopulation(
        startfreqs,
        config=config,
        generation=0,
        name='metapopulation'
    )
    # store initial state in database:
    rstore.record_special_state(metapop.generation, 'start')
    rstore.dump_data(metapop)
else:
    g,desc = special_states.pop()
    freqs,g = rstore.get_frequencies(g)
    metapop = core.MetaPopulation(
        freqs,
        config=config,
        generation=g,
        name='metapopulation'
    )

# <codecell>

print metapop
print metapop.overview()
print
fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)

# <markdowncell>

# Iterate until an equilibrium is reached:

# <codecell>

if not data_available:
    metapop.run(
        nmax,
        weights,
        thresh_total=eq,
        step=step,
        runstore=rstore,
        progress_bar=show_progressbar,
        verbose=True
    )
else:
    g,desc = special_states.pop()
    freqs,g = rstore.get_frequencies(g)
    metapop.set(g, freqs, desc)

# <headingcell level=3>

# 4.2 Migration-selection equilibrium

# <codecell>

print metapop
print metapop.overview()
print
fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)

# <headingcell level=3>

# 4.3 Introduction of preference alleles P3, P4, and P5

# <codecell>

if not data_available:
    intro_allele = 'P3'
    metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=False)
    rstore.dump_data(metapop)
    rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
    intro_allele = 'P5'
    metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=False)
    rstore.dump_data(metapop)
    rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
    
    intro_allele = 'P4'
    metapop.introduce_allele('pop2', intro_allele, intro_freq=intro, advance_generation_count=False)
    rstore.dump_data(metapop)
    rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
    intro_allele = 'P5'
    metapop.introduce_allele('pop2', intro_allele, intro_freq=intro, advance_generation_count=False)
    rstore.dump_data(metapop)
    rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
else:
    for i in range(4):   # once for each preference allele introduction
        g,desc = special_states.pop()
    freqs,g = rstore.get_frequencies(g)
    metapop.set(g, freqs, desc)

print metapop
print metapop.overview()

# <markdowncell>

# Iterate again until an equilibrium is reached:

# <codecell>

if not data_available:
    metapop.run(
        nmax,
        weights,
        thresh_total=eq,
        step=step,
        runstore=rstore,
        progress_bar=show_progressbar,
        verbose=True
    )
else:
    g,desc = special_states.pop()
    freqs,g = rstore.get_frequencies(g)
    metapop.set(g, freqs, desc)

# <headingcell level=3>

# 4.4 Final state

# <codecell>

print metapop
print metapop.overview()
print
fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)

# <headingcell level=4>

# Dynamic weights (in the final state)

# <codecell>

print TP

# <headingcell level=3>

# 4.5 Runtime

# <codecell>

rstore.flush()
if not data_available:
    print utils.timing_report(starttime, metapop.generation)

# <headingcell level=2>

# 5. Population dynamics

# <codecell>

fig = rstore.plot_sums(figsize=[max_figwidth, figheight])
show()

# <codecell>

rstore.close()

