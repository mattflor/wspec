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
LOCI = ['population', 'Alocus', 'Blocus', 'Clocus', 'Dlocus', 'trait', 'preference', 'cytotype']
ALLELES = [['pop1', 'pop2'], \
           ['A0', 'A1'], \
           ['B0', 'B1'], \
           ['C0', 'C1'], \
           ['D0', 'D1'], \
           ['T1', 'T2'], \
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
transition_probability = pt = 0.95   # probability of transition into another mating round
rejection_probability_trait1 = pr_t1 = 1.
rejection_probability_trait2 = pr_t2 = 1.    # probability to reject a non-preferred male
hybrid_male_sterility = h = 1.
introduction_frequency = intro = 0.05        # introduction frequency of preference mutant allele
threshold = 5e-3                   # equilibrium threshold
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
mig = np.array([[1-m,      m ], \
                [  m,    1-m ]], float)
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
viab = np.array([[ 1+s,    1  ], \
                 [   1,  1+s  ]], float)
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
weights['dynamic_reproduction'] = []
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

#! Hybrid male sterility due to B1 mutation
#!----------------------------------------------------------------------
HMS_AB, HMS_AB_ = core.hms_generator(('A',1), ('B',1), config, h)
print HMS_AB
print

#! Hybrid male sterility due to C1 mutation
#!----------------------------------------------------------------------
HMS_AC, HMS_AC_ = core.hms_generator(('A',0), ('C',1), config, h)
print HMS_AC
print

HMS_BC, HMS_BC_ = core.hms_generator(('B',1), ('C',1), config, h)
print HMS_BC
print

#! Hybrid male sterility due to D1 mutation
#!----------------------------------------------------------------------
HMS_AD, HMS_AD_ = core.hms_generator(('A',1), ('D',1), config, h)
print HMS_AD
print

HMS_BD, HMS_BD_ = core.hms_generator(('B',0), ('D',1), config, h)
print HMS_BD
print

HMS_CD, HMS_CD_ = core.hms_generator(('C',1), ('D',1), config, h)
print HMS_CD
print

#! Nuclear inheritance
#!----------------------------------------------------------------------
#! Preference locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
IP = core.ReproductionWeight(name='preference inheritance', \
                             axes=['female_preference', 'male_preference', 'offspring_preference'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('preference')])
IP.set( utils.nuclear_inheritance(n_alleles) )
IP_ = IP.extended()
print IP

#! Trait locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
IT = core.ReproductionWeight(name='trait inheritance', \
                             axes=['female_trait', 'male_trait', 'offspring_trait'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('trait')])
IT.set( utils.nuclear_inheritance(n_alleles) )
IT_ = IT.extended()
print IT

#! Background locus A
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
IA = core.ReproductionWeight(name='Alocus inheritance', \
                             axes=['female_Alocus', 'male_Alocus', 'offspring_Alocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Alocus')])
IA.set( utils.nuclear_inheritance(n_alleles) )
IA_ = IA.extended()
print IA

#! Background locus B
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
IB = core.ReproductionWeight(name='Blocus inheritance', \
                             axes=['female_Blocus', 'male_Blocus', 'offspring_Blocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Blocus')])
IB.set( utils.nuclear_inheritance(n_alleles) )
IB_ = IB.extended()
print IB

#! Background locus C
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
IC = core.ReproductionWeight(name='Clocus inheritance', \
                             axes=['female_Clocus', 'male_Clocus', 'offspring_Clocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Clocus')])
IC.set( utils.nuclear_inheritance(n_alleles) )
IC_ = IC.extended()
print IC

#! Background locus B
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be didplayed
ID = core.ReproductionWeight(name='Dlocus inheritance', \
                             axes=['female_Dlocus', 'male_Dlocus', 'offspring_Dlocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Dlocus')])
ID.set( utils.nuclear_inheritance(n_alleles) )
ID_ = ID.extended()
print ID

# we can combine all reproduction weights that are not frequency-dependent:
R_ = CI_ * F_ * T_ * \
     IP_ * IT_ * IA_ * IB_ * IC_ * ID_ * \
     HMS_AB_ * \
     HMS_AC_ * HMS_BC_ * \
     HMS_AD_ * HMS_BD_ * HMS_CD_
weights['constant_reproduction'] = R_

    
#! Simulation
#!======================================================================
desc = """
- 2 populations linked by symmetrical migration
- populations 1 uninfected, pop. 2 Wolbachia-infected
- a different trait adaptive in each population (T1 in pop.1, T2 in pop.2)
- hybrid males are fully sterile due to divergence at several loci (A, B, C, and D):
  Orr (1995): A1-B1, A0-C1, B1-C1, ... males are sterile
- allele P1 (female preference for trait T1) is introduced at equilibrium in pop.1
- after a new equilibrium has been reached, allele P2 (female preference for trait T2) is introduced in pop.2
- simulation is over when the final equilibrium has been reached
"""
snum = 16
rstore = storage.RunStore('/extra/flor/data/scenario_{0}.h5'.format(snum))
rnum = 2
try: rstore.select_scenario(snum, verbose=False)
except: rstore.create_scenario(snum, labels=(LOCI,ALLELES), description=desc)
try: rstore.remove_run(rnum, snum)
except: pass
rstore.init_run(rnum, parameters, FSHAPE, init_len=100)

#~ mode = None
mode = 'report'      # create a report with pyreport

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
startfreqs[0,1,0,1,0,0,0,0] = 1.                   # pop1-A1-B0-C1-D0-T1-P0-U
startfreqs[1,0,1,0,1,1,0,1] = 1.                   # pop2-A0-B1-C0-D1-T2-P0-W
starttime = time.time()
metapop = core.MetaPopulation(startfreqs, config=config, generation=0, name='metapopulation')
rstore.record_special_state(metapop.generation, 'start')
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
print metapop.overview(['Alocus', 'Blocus', 'Clocus', 'Dlocus'], ['trait', 'preference'], 'cytotype')

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
print metapop.overview(['Alocus', 'Blocus', 'Clocus', 'Dlocus'], ['trait', 'preference'], 'cytotype')

#! Introduction of preference allele P1
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
intro_allele = 'P1'
metapop.introduce_allele('pop1', intro_allele, intro_freq=intro, advance_generation_count=True)
rstore.dump_data(metapop)
rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview(['Alocus', 'Blocus', 'Clocus', 'Dlocus'], ['trait', 'preference'], 'cytotype')

#! P1 Equilibrium
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
print metapop.overview(['Alocus', 'Blocus', 'Clocus', 'Dlocus'], ['trait', 'preference'], 'cytotype')

#! Introduction of preference allele P2
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
intro_allele = 'P2'
metapop.introduce_allele('pop2', intro_allele, intro_freq=intro, advance_generation_count=True)
rstore.dump_data(metapop)
rstore.record_special_state(metapop.generation, 'intro {0}'.format(intro_allele))
print metapop
#! Locus overview
#!----------------------------------------------------------------------
print metapop.overview(['Alocus', 'Blocus', 'Clocus', 'Dlocus'], ['trait', 'preference'], 'cytotype')

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
print metapop.overview(['Alocus', 'Blocus', 'Clocus', 'Dlocus'], ['trait', 'preference'], 'cytotype')
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
    print 'Simulation run completed:'
    seconds = time.time()-starttime
    hhmmss = str(datetime.timedelta(seconds=int(seconds)))
    print 'Generation: {0} (Elapsed Time: {1})'.format(metapop.generation, hhmmss)
rstore.flush()

#! Dynamics
#!======================================================================
fig = rstore.plot_sums(figsize=figsize)
figs.append(fig)
show()

rstore.close()
