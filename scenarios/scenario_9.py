import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint, datetime
sys.path.append(".")             # pyreport needs this to know where to import modules from
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from pylab import show, close          # pyreport needs this to find figures
import core, storage
import visualization as viz
import utilities as utils
for mod in [core,storage,utils,viz]:
    reload(mod)
np.set_printoptions(precision=4, suppress=True, linewidth=200)


#! .. contents:: :depth: 5

#!
#! .. sectnum:: :depth: 5
#!
#$ \newpage
#! Configuration
#!======================================================================
#! Loci and alleles
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LOCI = ['population', 'trait', 'preference', 'cytotype']
ALLELES = [['pop1', 'pop2', 'pop3', 'pop4'], \
           ['T1', 'T2', 'T3', 'T4'], \
           ['P0', 'P1', 'P2'], \
           ['U', 'W']
          ]
loc_width = len(max(LOCI, key=len))
for i,loc in enumerate(LOCI):
    print "%-*s  :\t%s" % (loc_width, loc, ', '.join(ALLELES[i]))
    
#! Parameters
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
migration_rate = m = 0.01           # symmetric migration!
selection_coefficient = s = 1.      # T1 in pop1, T2 in pop2, etc.
ci_level = l = 0.9                  # CI level
fecundity_reduction = f = 0.        # fecundity reduction in infected females
transmission_rate = t = 0.87        # transmission of Wolbachia
transition_probability = pt = 0.99   # probability of transition into another mating round
rejection_probability_trait1 = pr_t1 = 1.
rejection_probability_trait2 = pr_t2 = 1.    # probability to reject a non-preferred male
introduction_frequency = intro = 0.05        # introduction frequency of preference mutant allele
threshold = 1e-4                   # equilibrium threshold
parameters = dict(m=m, s=s, lCI=l, f=f, t=t, pt=pt, pr_t1=pr_t1, pr_t2=pr_t2, intro=intro, threshold=threshold)           # dictionary for storing simulation
par_width = len(max(parameters.keys(), key=len))
for p,v in sorted(parameters.items()):
    print "%-*s  :\t%s" % (par_width, p, v)
print


# setting up scenario config
def configure():
    config = {}
    config['LOCI'] = LOCI
    config['ALLELES'] = ALLELES
    config['ADICT'] = utils.make_allele_dictionary(LOCI, ALLELES)
    #~ config['LABELS'] = panda_index(ALLELES, LOCI)   # use this as index for conversion of freqs to pd.Series
    config['FSHAPE'] = utils.list_shape(ALLELES)          # shape of frequencies
    repro_axes = utils.reproduction_axes(LOCI)
    config['REPRO_AXES'] = repro_axes  # axes for the reproduction step, used for automatic extension of arrays to the correct shape by inserting np.newaxis
    config['N_LOCI'] = len(LOCI)
    pops = ALLELES[0]
    config['POPULATIONS'] = pops              # shortcut for faster access to populations
    config['N_POPS'] = len(pops)             # number of populations within metapopulations
    config['REPRO_DIM'] = len(repro_axes)
    return config
config = configure()           # dictionary for storing simulation
locals().update(config)

        
#! Weights
#!======================================================================
weights = {}           # dictionary for storing simulation

#! Migration
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                   
#~ # pop1, pop2, and pop3 are of equal size, pop4 is twice as large:
#~ mig = np.array([[1-m,       m,      0,      0], \
                #~ [  m,   1-2*m,      m,      0], \
                #~ [  0,       m,  1-2*m,      m/2], \
                #~ [  0,       0,      m,    1-m/2]], float)
mig = np.array([[1-m,       m,      0,      0], \
                [  m,   1-2*m,      m,      0], \
                [  0,       m,  1-2*m,      m], \
                [  0,       0,      m,    1-m]], float)
M = core.MigrationWeight(name='migration', \
                         axes=['target', 'source'], \
                         config=config, \
                         arr=mig, \
                         m=migration_rate
                        )
weights['migration'] = M.extended()
print M

#! Viability selection
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
viab = np.array([[ 1+s,    1,    1,    1  ], \
                 [   1,  1+s,    1,    1  ], \
                 [   1,    1,  1+s,    1  ], \
                 [   1,    1,    1,  1+s  ]], float)
VS = core.ViabilityWeight(name='viability selection', \
                          axes=['population','trait'], \
                          config=config, \
                          arr=viab, \
                          s=selection_coefficient
                         )
weights['viability_selection'] = VS.extended()
print VS

#! Sexual selection (female mating preference)
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#! These weights are frequency-dependent. Their final states can be found
#! in section `Dynamic weights (final states)`_.
#!
weights['dynamic_reproduction'] = []    # list of tuples (weight, [target_loci])
#! Trait preference
#!----------------------------------------------------------------------
#~ trait_preferences = {'P1': {'all pops': ('T3', pr_t1)}, \
                     #~ 'P2': {'all pops': ('T4', pr_t2)}
                    #~ }
#~ TP = core.PreferenceWeight(
    #~ name='trait preference', \
    #~ axes=['population', 'female_preference', 'male_trait'], \
    #~ pref_desc = trait_preferences, \
    #~ config=config, \
    #~ unstack_levels=[2], \
    #~ pt=transition_probability, \
    #~ pr_t1=rejection_probability_trait1,
    #~ pr_t2=rejection_probability_trait2
    #~ )
trait_preferences = {
    'P0': {'baseline': 0.}, \
    'P1': {'baseline': pr_t1, 'T3': 1-pr_t1},   # all traits not explicitely mentioned will be rejected with the baseline probability
    'P2': {'baseline': pr_t2, 'T4': 1-pr_t2}
    }
TP = core.GeneralizedPreferenceWeight(
    name='trait preference', \
    axes=['population', 'female_preference', 'male_trait'], \
    pref_desc = trait_preferences, \
    config=config, \
    unstack_levels=[2], \
    pt=transition_probability, \
    )
weights['dynamic_reproduction'].append( (TP, ['trait']) )
print TP

#! Reproduction
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#! Cytoplasmic incompatibility
#!----------------------------------------------------------------------
CI = core.ReproductionWeight(name='cytoplasmic incompatibility', \
                             axes=['male_cytotype', 'offspring_cytotype'], \
                             config=config, \
                             unstack_levels=[1], \
                             lCI=ci_level
                            )
CI.set( np.array([[1,1],[1-ci_level,1]],float) )
CI_ = CI.extended()
print CI

#! Female fecundity
#!----------------------------------------------------------------------
F = core.ReproductionWeight(name='fecundity reduction', \
                            axes=['female_cytotype'], \
                            config=config, \
                            f=fecundity_reduction
                           )
F.set( np.array([1,1-f],float) )
F_ = F.extended()
print F

#! Cytotype inheritance (Wolbachia transmission)
#!----------------------------------------------------------------------
T = core.ReproductionWeight(name='cytotype inheritance', \
                            axes=['female_cytotype', 'offspring_cytotype'], \
                            config=config, \
                            unstack_levels=[1], \
                            t=transmission_rate
                           )
T.set( np.array([[1,0],[1-t,t]],float) )
T_ = T.extended()
print T

#! Nuclear inheritance
#!----------------------------------------------------------------------
#! Preference locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
PI = core.ReproductionWeight(name='preference inheritance', \
                             axes=['female_preference', 'male_preference', 'offspring_preference'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('preference')])
PI.set( utils.nuclear_inheritance(n_alleles) )
PI_ = PI.extended()
print PI

#! Trait locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
TI = core.ReproductionWeight(name='trait inheritance', \
                             axes=['female_trait', 'male_trait', 'offspring_trait'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('trait')])
TI.set( utils.nuclear_inheritance(n_alleles) )
TI_ = TI.extended()
print TI

# we can combine all reproduction weights that are not frequency-dependent:
R_ = CI_ * F_ * T_ * PI_ * TI_
weights['constant_reproduction'] = R_

    
#! Simulation
#!======================================================================
desc = """
- 4 populations, arranged like stepping stones linked by symmetrical migration
- populations 1-3 representing uninfected D. sub.; pop.4 representing Wolbachia-infected D. rec. (initially)
- a different trait adaptive in each population (T1 in pop.1, T2 in pop.2, T3 in pop.3, T4 in pop.4)
- allele P1 (female preference for trait T3) is introduced at equilibrium in pop.3
- after a new equilibrium has been reached, allele P2 (female preference for trait T4) is introduced in pop.4
- simulation is over when the final equilibrium has been reached
"""
#~ snum = 9
#~ rstore = storage.RunStore('/extra/flor/data/scenario_{0}.h5'.format(snum))
#~ rnum = 3
#~ try: rstore.select_scenario(snum, verbose=False)
#~ except: rstore.create_scenario(snum, labels=(LOCI,ALLELES), description=desc)
#~ try: rstore.remove_run(rnum, snum)
#~ except: pass
#~ rstore.init_run(rnum, parameters, FSHAPE, init_len=100)
#~ 
mode = None
#~ mode = 'report'      # create a report with pyreport
#~ 
if mode == 'report':
    progress = False
else:
    progress = True
#~ 
n = 50
step = 10
#~ figs = []
#~ figsize = [20,11]

#! Start
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
startfreqs = np.zeros(FSHAPE)
startfreqs[0,0,0,0] = 1.                   # pop1-T1-P0-U
startfreqs[1,1,0,0] = 1.                   # pop2-T2-P0-U
startfreqs[2,2,0,0] = 1.                   # pop3-T3-P0-U
startfreqs[3,3,0,1] = 1.                   # pop4-T4-P0-W
starttime = time.time()
metapop = core.MetaPopulation(startfreqs, config=config, generation=0, name='metapopulation')
#~ rstore.record_special_state(metapop.generation, 'start')
#~ rstore.dump_data(metapop)
#~ #! Plot
#~ #!----------------------------------------------------------------------
#~ fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
#~ figs.append(fig)
#~ show()
#~ #! Nucleocytotype frequencies
#~ #!----------------------------------------------------------------------
#~ print metapop
#~ #! Locus overview
#~ #!----------------------------------------------------------------------
#~ print metapop.overview()
#~ 
#~ #! Migration-selection equilibrium
#~ #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~ progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
progress = metapop.run(n, weights, threshold=threshold, step=step, progress=progress)
#~ #! Plot
#~ #!----------------------------------------------------------------------
#~ fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
#~ figs.append(fig)
#~ show()
#~ #! Nucleocytotype frequencies
#~ #!----------------------------------------------------------------------
#~ print metapop
#~ #! Locus overview
#~ #!----------------------------------------------------------------------
#~ print metapop.overview()
#~ 
#~ #! Introduction of preference allele P1
#~ #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~ intro_allele = 'P1'
#~ metapop.introduce_allele('pop3', intro_allele, intro_freq=intro, advance_generation_count=True)
#~ rstore.dump_data(metapop)
#~ rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
#~ print metapop
#~ #! Locus overview
#~ #!----------------------------------------------------------------------
#~ print metapop.overview()
#~ 
#~ 
#~ #! P1 Equilibrium
#~ #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~ progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
#~ #! Plot
#~ #!----------------------------------------------------------------------
#~ fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
#~ figs.append(fig)
#~ show()
#~ #! Nucleocytotype frequencies
#~ #!----------------------------------------------------------------------
#~ print metapop
#~ #! Locus overview
#~ #!----------------------------------------------------------------------
#~ print metapop.overview()
#~ 
#~ #! Introduction of preference allele P2
#~ #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~ intro_allele = 'P2'
#~ metapop.introduce_allele('pop4', intro_allele, intro_freq=intro, advance_generation_count=True)
#~ rstore.dump_data(metapop)
#~ rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
#~ print metapop
#~ #! Locus overview
#~ #!----------------------------------------------------------------------
#~ print metapop.overview()
#~ 
#~ #! Final
#~ #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~ progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
#~ #! Plot
#~ #!----------------------------------------------------------------------
#~ fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
#~ figs.append(fig)
#~ show()
#~ #! Nucleocytotype frequencies
#~ #!----------------------------------------------------------------------
#~ print metapop
#~ #! Locus overview
#~ #!----------------------------------------------------------------------
#~ print metapop.overview()
#~ #! Runtime
#~ #!----------------------------------------------------------------------
#~ if mode == 'report':
    #~ print 'Simulation run completed:'
    #~ seconds = time.time()-starttime
    #~ hhmmss = str(datetime.timedelta(seconds=int(seconds)))
    #~ print 'Generation: {0} (Elapsed time: {1})'.format(metapop.generation, hhmmss)
    #~ print ' '
    #~ 
#~ #! Dynamic weights (final states)
#~ #!======================================================================
#~ 
#~ #! Sexual selection (final)
#~ #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~ 
#~ #! Trait preference (final)
#~ #!----------------------------------------------------------------------
#~ print TP
#~ 
#~ if not mode == 'report':
    #~ print 'Simulation run completed:'
    #~ seconds = time.time()-starttime
    #~ hhmmss = str(datetime.timedelta(seconds=int(seconds)))
    #~ print 'Generation: {0} (Elapsed Time: {1})'.format(metapop.generation, hhmmss)
#~ rstore.flush()
#~ 
#~ #! Dynamics
#~ #!======================================================================
#~ fig = rstore.plot_sums(figsize=figsize)
#~ figs.append(fig)
#~ show()
#~ 
#~ rstore.close()
