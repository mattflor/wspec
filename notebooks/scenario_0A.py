# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Scenario 0

# <codecell>

import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint, datetime, pdb
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import Image
# wspec moduls:
import core, storage
import visualization as viz
import utilities as utils
for mod in [core,storage,utils,viz]:     # reload the wspec modules in case the code has changed
    reload(mod)
    
np.set_printoptions(precision=4, suppress=True, linewidth=100)

# <markdowncell>

# This scenario has the following features:
# 
# * single population
# * a neutral trait T0 and a preference allele P0 (non-discriminating) are fixed
# * a new, adaptive trait T1 is introduced
# * a preference for this trait is introduced: P1 (T1); this may happen after the introduction of T1 or simultaneously with it
# 
# <!-- <img src="files/images/setup_01.png">  -->
# <img src="https://docs.google.com/drawings/d/soLzuzqLT2AcRMbaZRpfUxg/image?w=153&h=248&rev=298&ac=1">

# <markdowncell>

# ## Configuration
# 
# To configure the simulation scenario, we need to specify gene loci, alleles, and parameters.
# 
# ### Loci and alleles
# 
# Populations are treated as loci, as would be the cytotype. In numpy terms, each locus is represented by an array axis.
# 
# We keep separate lists rather than a dictionary because we need to preserve the locus and allele orders.

# <codecell>

LOCI = ['population', 'trait', 'preference']
ALLELES = [['pop1'],
           ['T0', 'T1'],
           ['P0', 'P1']
          ]
print utils.loci2string(LOCI, ALLELES)

# <headingcell level=3>

# Parameters

# <codecell>

sid = 0     # scenario id
rid = 'A'     # id of simulation run

# <codecell>

PARAMETERS = {
    's': (0.1, 'selection coefficient'),         # selection advantage for adaptive trait
    'pt': (0.9, 'transition probability'),       # probability of transition into another mating round
    'intro': (0.001, 'introduction frequency'), # introduction frequency of preference mutant allele
    'eq': (1e-6, 'equilibrium threshold'),      # equilibrium threshold (total frequency change)
    'nmax': (30000, 'max generation'),          # max number of generations to iterate for each stage of the simulation
    'step': (10, 'storage stepsize')            # store metapopulation state every `step` generations
}
# For mating preference parameters, we use a different notation:
trait_preferences = {                        # female mating preferences (rejection probabilities)
    'P0': {'baseline': 0.},
    'P1': {'baseline': 0.4, 'T1': 0.}
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

# Weights

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

# Viability selection

# <codecell>

vsarr = np.array(
    [[   1,  1+s]], float
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

# Sexual selection (female mating preference)

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

# Reproduction

# <headingcell level=4>

# Nuclear inheritance

# <headingcell level=5>

# Preference locus

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

# <headingcell level=5>

# Trait locus

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

R_ = IP_ * IT_
weights['constant_reproduction'] = R_

# <headingcell level=2>

# Simulation

# <headingcell level=3>

# Initial state

# <codecell>

if not data_available:
    starttime = time.time()                  # take time for timing report after simulation run
    startfreqs = np.zeros(FSHAPE)
    startfreqs[0,1,0] = 1.                   # pop1-T0-P0
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
fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)

# <markdowncell>

# Run the simulation until an equilibrium is reached (but for `nmax` generations at most):

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

# Equilibrium

# <codecell>

print metapop
print metapop.overview()
fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)

# <headingcell level=3>

# Introduction of trait allele T1

# <codecell>

if not data_available:
    #intro_allele = 'T1'
    #metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=False)
    #rstore.dump_data(metapop)
    #rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
    
    intro_allele = 'P1'
    metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=True)
    rstore.dump_data(metapop)
    rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
else:
    for i in range(1):   # once for each preference allele introduction
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

# Final state

# <codecell>

print metapop
print metapop.overview()
fig = viz.plot_overview(metapop, show_generation=False, figsize=figsize)

# <headingcell level=3>

# Dynamic weights (final states)

# <codecell>

print TP

# <headingcell level=3>

# Runtime

# <codecell>

rstore.flush()
if not data_available:
    print utils.timing_report(starttime, metapop.generation)

# <headingcell level=2>

# Population dynamics

# <codecell>

fig = rstore.plot_sums(figsize=[max_figwidth, figheight])
show()

# <codecell>

rstore.close()

