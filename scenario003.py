import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint
sys.path.append(".")             # pyreport needs this to know where to import modules from
import numpy as np
import numpy.random as npr
import pandas as pd
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
LOCI = ['population', 'backA', 'backB', 'recognition', 'trait', 'preference', 'cytotype']
ALLELES = [['pop1','pop2','pop3','pop4'], \
           ['A1', 'A2'], \
           ['B1', 'B2'], \
           ['S1', 'S2'], \
           ['T1','T2',], \
           ['P0', 'P1', 'P2'], \
           ['U', 'W']
          ]
loc_width = len(max(LOCI, key=len))
for i,loc in enumerate(LOCI):
    print "%-*s  :\t%s" % (loc_width, loc, ', '.join(ALLELES[i]))
    
#! Parameters
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
migration_rate = m = 0.01           # symmetric migration!
selection_coefficient = s = 1.     # T1 in pop1, T2 in pop2
ci_level = l = 0.9                  # CI level
fecundity_reduction = f = 0.        # fecundity reduction in infected females
transmission_rate = t = 0.87        # transmission of Wolbachia
transition_probability = pt = 0.95   # probability of transition into another mating round
rejection_probability_species1 = pr_s1 = 0.7
rejection_probability_species2 = pr_s2 = 0.9
rejection_probability_trait1 = pr_t1 = 1.
rejection_probability_trait2 = pr_t2 = 1.    # probability to reject a non-preferred male
hybrid_male_sterility = h = 1.
introduction_frequency = intro = 0.05        # introduction frequency of preference mutant allele
threshold = 5e-3                   # equilibrium threshold
parameters = dict(m=m, s=s, lCI=l, f=f, t=t, pt=pt, pr_s1=pr_s1, pr_s2=pr_s2, pr_t1=pr_t1, pr_t2=pr_t2, intro=intro, threshold=threshold)           # dictionary for storing simulation
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
viab = np.array([[   1+s,        1    ], \
                 [   1+s,        1    ], \
                 [   1+s,        1    ], \
                 [     1,      1+s    ]], float)
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
#! Species recognition (preference for background loci)
#!----------------------------------------------------------------------
species_recognition = {'S1': {'all pops': ('A1-B1', pr_s1)}, \
                       'S2': {'all pops': ('A2-B2', pr_s2)}
                      }
pprint.pprint(species_recognition)
SR = core.PreferenceWeight(name='species recognition', \
                           axes=['population', 'female_recognition', 'male_backA', 'male_backB'], \
                           pref_desc = species_recognition, \
                           config=config, \
                           unstack_levels=[3], \
                           pt=transition_probability, \
                           pr_s1=rejection_probability_species1,
                           pr_s2=rejection_probability_species2
                          )
weights['dynamic_reproduction'] = [SR]
print SR

#! Trait preference
#!----------------------------------------------------------------------
trait_preferences = {'P1': {'all pops': ('T1', pr_t1)}, \
                     'P2': {'all pops': ('T2', pr_t2)}
                    }
TP = core.PreferenceWeight(name='trait preference', \
                           axes=['population', 'female_preference', 'male_trait'], \
                           pref_desc = trait_preferences, \
                           config=config, \
                           unstack_levels=[2], \
                           pt=transition_probability, \
                           pr_t1=rejection_probability_trait1,
                           pr_t2=rejection_probability_trait2
                          )
weights['dynamic_reproduction'].append(TP)
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

#! Hybrid male sterility
#!----------------------------------------------------------------------
HMS = core.ReproductionWeight(name='hybrid male sterility', \
                              axes=['male_backA', 'male_backB'], \
                              config=config, \
                              unstack_levels=[1], \
                              h=hybrid_male_sterility
                             )
HMS.set( np.array( [[1,1-h],[1-h,1]],float ) )
HMS_ = HMS.extended()
print HMS

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
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty
PI = core.ReproductionWeight(name='preference inheritance', \
                             axes=['female_preference', 'male_preference', 'offspring_preference'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
PI.set( utils.nuclear_inheritance(3) )
PI_ = PI.extended()
print PI

#! Trait locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty
TI = core.ReproductionWeight(name='trait inheritance', \
                             axes=['female_trait', 'male_trait', 'offspring_trait'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
TI.set( utils.nuclear_inheritance(2) )
TI_ = TI.extended()
print TI

#! Background locus A
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty
AI = core.ReproductionWeight(name='background A inheritance', \
                             axes=['female_backA', 'male_backA', 'offspring_backA'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
AI.set( utils.nuclear_inheritance(2) )
AI_ = AI.extended()
print AI

#! Background locus B
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty
BI = core.ReproductionWeight(name='background B inheritance', \
                             axes=['female_backB', 'male_backB', 'offspring_backB'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
BI.set( utils.nuclear_inheritance(2) )
BI_ = BI.extended()
print BI

#! Species recognition locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty
SI = core.ReproductionWeight(name='species recognition inheritance', \
                             axes=['female_recognition', 'male_recognition', 'offspring_recognition'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
SI.set( utils.nuclear_inheritance(2) )
SI_ = SI.extended()
print SI

# we can combine all reproduction weights that are not frequency-dependent:
R_ = CI_ * F_ * T_ * PI_ * TI_ * AI_ * BI_ * SI_ * HMS_
weights['constant_reproduction'] = R_

    
#! Simulation
#!======================================================================
rstore = storage.RunStore('/extra/flor/data/simdata.h5')
#~ rstore = storage.Runstore('simdata2.h5')
snum = 5
rnum = 1
#~ rstore.select_scenario(snum)
#~ rstore.select_run(rnum)
try: rstore.select_scenario(snum)
except: rstore.create_scenario(snum, labels=(LOCI,ALLELES))
try: rstore.remove_run(rnum, snum)
except: pass
rstore.init_run(rnum, parameters, FSHAPE, init_len=100)

mode = None
#~ mode = 'report'      # create a report with pyreport

n = 50
step = 10

#! Start frequencies
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
startfreqs = np.zeros(FSHAPE)
startfreqs[0,0,0,0,0,0,0] = 1.                   # pop1-A1-B1-S1-T1-P0-U
startfreqs[1,0,0,0,0,0,0] = 1.                   # pop2-A1-B1-S1-T2-P0-U
startfreqs[2,0,0,0,0,0,0] = 1.                   # pop3-A1-B1-S1-T3-P0-U
startfreqs[3,1,1,1,1,0,1] = 1.                   # pop4-A2-B2-S2-T4-P0-W
metapop = core.MetaPopulation(startfreqs, config=config, generation=0, name='metapopulation')
print metapop

##! Migration-selection equilibrium
##!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore)
print metapop

##! Introduction of preference allele P1
##!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.introduce_allele('pop3', 'P1', intro_freq=intro, advance_generation_count=True)
rstore.dump_data(metapop)
print metapop

##! Equilibrium
##!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
print metapop

##! Introduction of preference allele P2
##!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.introduce_allele('pop4', 'P2', intro_freq=intro, advance_generation_count=True)
rstore.dump_data(metapop)
print metapop

##! Final state
##!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
progress_final = progress.finish(maxval=False, fdwrite=False)   # capture progress._format_line for printing at the very end
print metapop


##! Loci (sums)
##!----------------------------------------------------------------------
print metapop.overview()

    
##! Dynamic weights (final states)
##!======================================================================
##! Sexual selection (final)
##!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##! Species recognition (final)
##!----------------------------------------------------------------------
print SR

##! Trait preference (final)
##!----------------------------------------------------------------------
print TP

print
print progress_final
rstore.flush()

#! Plots
#!======================================================================
try:
    figs = rstore.plot_sums()
    bars = viz.stacked_bars(metapop.all_sums(), metapop.loci, metapop.alleles)
except:
    raise
if mode == 'report':
    show()
    close('all')

rstore.close()
