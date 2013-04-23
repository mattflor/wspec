# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Scenario 1

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
    
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# <markdowncell>

# This scenario has the following features:
# 
# * single population
# * a neutral trait T0 and a preference allele P0 (non-discriminating) are fixed
# * a new, adaptive trait T1 is introduced
# * a preference for this trait is introduced: P1 (T1); this may happend after the introduction of T1 or simultaneosuly with it
# 
# <img src="files/images/setup_01.png">

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
ALLELES = [['pop1', 'pop2'], \
           ['T1', 'T2'], \
           ['P1', 'P2']
          ]
print utils.loci2string(LOCI, ALLELES)

# <headingcell level=3>

# Parameters

# <codecell>

PARAMETERS = {
    's': (1., 'selection coefficient'),           # selection advantage for adaptive trait
    'pt': (0.95, 'transition probability'),       # probability of transition into another mating round
    'intro': (0.05, 'introduction frequency'),    # introduction frequency of preference mutant allele
    'eq': (5e-3, 'equilibrium threshold')                     # equilibrium threshold
}
# For mating preference parameters, we use a different notation:
trait_preferences = {                        # female mating preferences (rejection probabilities)
    'P1': {'baseline': 0.5, 'T1': 0.}, \
    'P2': {'baseline': 0.5, 'T2': 0.}
}
PARAMETERS = utils.add_preferences(PARAMETERS, trait_preferences)
print utils.params2string(PARAMETERS)

# <markdowncell>

# Update local variables so we can directly use loci, alleles, and parameters:

# <codecell>

config = utils.configure_locals(LOCI, ALLELES, PARAMETERS)
locals().update(config)
# pprint.pprint( sorted(config.items()) )

# <headingcell level=2>

# Weights

# <markdowncell>

# All weights for the different stages of the simulation are stored in a dictionary.

# <codecell>

weights = {
    'migration': None, \
    'viability_selection': None, \
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
    [[   1,  1+s], \
     [ 1+s,    1]], float
)
VS = core.ViabilityWeight(
    name='viability selection', \
    axes=['population','trait'], \
    config=config, \
    arr=vsarr, \
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
    name='trait preference', \
    axes=['population', 'female_preference', 'male_trait'], \
    pref_desc = trait_preferences, \
    config=config, \
    unstack_levels=[2], \
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
    name='preference inheritance', \
    axes=['female_preference', 'male_preference', 'offspring_preference'], \
    config=config, \
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
    name='trait inheritance', \
    axes=['female_trait', 'male_trait', 'offspring_trait'], \
    config=config, \
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

# <codecell>

snum = 38
rstore = storage.RunStore('/extra/flor/data/scenario_{0}.h5'.format(snum))
rnum = 1
try: rstore.select_scenario(snum, verbose=False)
except: rstore.create_scenario(snum, labels=(LOCI,ALLELES), description=desc)
try: rstore.remove_run(rnum, snum)
except: pass
rstore.init_run(rnum, parameters, FSHAPE, init_len=100)

#~ mode = None
mode = 'report'      # create a report with pyreport
def print_report():
    print 'Simulation run completed:'
    seconds = time.time()-starttime
    hhmmss = str(datetime.timedelta(seconds=int(seconds)))
    print 'Generation: {0}\nElapsed Time: {1}'.format(metapop.generation, hhmmss)
    pergen = seconds / metapop.generation
    hhmmss = str(datetime.timedelta(seconds=int(pergen)))
    print 'Time per generation: {0})'.format(hhmmss)

if mode == 'report':
    progress = False
else:
    progress = True

n = 20000
step = 10
figs = []
figsize = [20,11]

#! Start
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
startfreqs = np.zeros(FSHAPE)
startfreqs[0,0,0] = 1.                   # pop1-T1-P1
starttime = time.time()
metapop = core.MetaPopulation(startfreqs, config=config, generation=0, name='metapopulation')
rstore.record_special_state(metapop.generation, 'start')
#~ pdb.set_trace()
rstore.dump_data(metapop)
#! Plot
#!----------------------------------------------------------------------
fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
figs.append(fig)
show()
#! Nucleocytotype frequencies
#!----------------------------------------------------------------------
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview()

#! Migration-selection equilibrium
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
#! Plot
#!----------------------------------------------------------------------
fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
figs.append(fig)
show()
#! Nucleocytotype frequencies
#!----------------------------------------------------------------------
print metapop
#! Locus overview
#!----------------------------------------------------------------------

#! Introduction of trait allele T2
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
intro_allele = 'T2'
metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=True)
rstore.dump_data(metapop)
rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview()


#! Equilibrium
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
#! Plot
#!----------------------------------------------------------------------
fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
figs.append(fig)
show()
#! Nucleocytotype frequencies
#!----------------------------------------------------------------------
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview()


#! Introduction of preference allele P2
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
intro_allele = 'P2'
metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=True)
rstore.dump_data(metapop)
rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview()


#! Final
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
#! Plot
#!----------------------------------------------------------------------
fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
figs.append(fig)
show()
#! Nucleocytotype frequencies
#!----------------------------------------------------------------------
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview()

#! Runtime
#!----------------------------------------------------------------------
if mode == 'report':
    print 'Simulation run completed:'
    seconds = time.time()-starttime
    hhmmss = str(datetime.timedelta(seconds=int(seconds)))
    print 'Generation: {0} (Elapsed time: {1})'.format(metapop.generation, hhmmss)
    print ' '
    
#! Dynamic weights (final states)
#!======================================================================
#! Sexual selection (final)
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#! Trait preference (final)
#!----------------------------------------------------------------------
print TP

if not mode == 'report':
    print_report()
rstore.flush()

#! Dynamics
#!======================================================================
fig = rstore.plot_sums(figsize=figsize)
figs.append(fig)
show()

rstore.close()

