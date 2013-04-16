import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint, datetime, math
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
LOCI = ['population', \
        'Alocus', 'Blocus', \
        'Clocus', 'Dlocus', \
        'Elocus', 'Flocus', \
        'trait', 'cytotype']
ALLELES = [['pop1', 'pop2'], \
           ['A0', 'A1'], \
           ['B0', 'B1'], \
           ['C0', 'C1'], \
           ['D0', 'D1'], \
           ['E0', 'E1'], \
           ['F0', 'F1'], \
           ['T1', 'T2'], \
           ['U', 'W']
          ]
loc_width = len(max(LOCI, key=len))
for i,loc in enumerate(LOCI):
    print "%-*s  :\t%s" % (loc_width, loc, ', '.join(ALLELES[i]))
    
#! Parameters (without migration)
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
selection_coefficient = s = 1.      # T1 in pop1, T2 in pop2, etc.
ci_level = l = 0.9                  # CI level
fecundity_reduction = f = 0.        # fecundity reduction in infected females
transmission_rate = t = 0.87        # transmission of Wolbachia
hybrid_male_sterility = h = 1.
threshold = 5e-3                    # equilibrium threshold
wolb_prec = 0.1                     # precicsion for wolbachia infection pattern stability
hms_prec = 0.5                     # precicsion for HMS pattern stability
parameters = dict(s=s, lCI=l, f=f, t=t, threshold=threshold, hms_prec=hms_prec)           # dictionary for storing simulation
par_width = len(max(parameters.keys(), key=len))
for p,v in sorted(parameters.items()):
    print "%-*s  :\t%s" % (par_width, p, v)
print                

import analytical
mcrit = \
    {'analytical': \
        {'uninfected mainland': analytical.mcrit_UMA(f=f, ci=l, s=s, t=t), \
         'infected mainland':   analytical.mcrit_IMA(f=f, ci=l, s=s, t=t)}, \
     'numerical': \
        {'uninfected mainland': 'to be determined', \
         'infected mainland':   'to be determined'}
    }
def wolb_infect_stable(metapop, prec=0.1):
    """
    Return 1 if Wolbachia infection pattern is stable and 0 if it is not.
    """
    cfreqs = metapop.get_sums('cytotype')
    return abs(cfreqs[0,1] - cfreqs[1,1]) > prec

def hms_stable(metapop, prec=0.1):
    """
    Return 1 if HMS pattern is stable and 0 if it is not.
    """
    afreqs = metapop.get_sums('Alocus')
    bfreqs = metapop.get_sums('Blocus')
    return ( abs(afreqs[0,1] - afreqs[1,1]) > prec ) and ( abs(bfreqs[0,1] - bfreqs[1,1]) > prec )

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

#! Weights (same for all runs)
#!======================================================================
weights = {}           # dictionary for storing simulation

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

def hms_generator((locus1, allele1), (locus2, allele2), h=hybrid_male_sterility):
    """
    Usage: hms_generator(('A',1), ('B',0)) generates a weight for HMS
           due to incompatibilities between the 'Alocus' and the 'Blocus'
           with hybrid males carrying the allele combination 'A1-B0'
           being sterile
    """
    HMS_weight = core.ReproductionWeight(name='hybrid male sterility {0}/{1}'.format(locus1,locus2), \
        axes=['male_{0}locus'.format(locus1), 'male_{0}locus'.format(locus2)], \
        config=config, \
        unstack_levels=[1], \
        h=h
        )
    n1 = len(ALLELES[LOCI.index('{0}locus'.format(locus1))])
    n2 = len(ALLELES[LOCI.index('{0}locus'.format(locus2))])
    ary = np.ones((n1,n2), float)
    ary[allele1,allele2] = 1-h
    HMS_weight.set( ary )
    HMS_weight_ = HMS_weight.extended()
    return HMS_weight, HMS_weight_
                            
#! Hybrid male sterility due to B1 mutation
#!----------------------------------------------------------------------
HMS_AB, HMS_AB_ = hms_generator(('A',1), ('B',1), h)
print HMS_AB
print

#! Hybrid male sterility due to C1 mutation
#!----------------------------------------------------------------------
HMS_AC, HMS_AC_ = hms_generator(('A',0), ('C',1), h)
print HMS_AC
print

HMS_BC, HMS_BC_ = hms_generator(('B',1), ('C',1), h)
print HMS_BC
print

#! Hybrid male sterility due to D1 mutation
#!----------------------------------------------------------------------
HMS_AD, HMS_AD_ = hms_generator(('A',1), ('D',1), h)
print HMS_AD
print

HMS_BD, HMS_BD_ = hms_generator(('B',0), ('D',1), h)
print HMS_BD
print

HMS_CD, HMS_CD_ = hms_generator(('C',1), ('D',1), h)
print HMS_CD
print

#! Hybrid male sterility due to E1 mutation
#!----------------------------------------------------------------------
HMS_AE, HMS_AE_ = hms_generator(('A',0), ('E',1), h)
print HMS_AE
print

HMS_BE, HMS_BE_ = hms_generator(('B',1), ('E',1), h)
print HMS_BE
print

HMS_CE, HMS_CE_ = hms_generator(('C',0), ('E',1), h)
print HMS_CE
print

HMS_DE, HMS_DE_ = hms_generator(('D',1), ('E',1), h)
print HMS_DE
print

#! Hybrid male sterility due to F1 mutation
#!----------------------------------------------------------------------
HMS_AF, HMS_AF_ = hms_generator(('A',1), ('F',1), h)
print HMS_AF
print

HMS_BF, HMS_BF_ = hms_generator(('B',0), ('F',1), h)
print HMS_BF
print

HMS_CF, HMS_CF_ = hms_generator(('C',1), ('F',1), h)
print HMS_CF
print

HMS_DF, HMS_DF_ = hms_generator(('D',0), ('F',1), h)
print HMS_DF
print

HMS_EF, HMS_EF_ = hms_generator(('E',1), ('F',1), h)
print HMS_EF
print


#! Nuclear inheritance
#!----------------------------------------------------------------------
#! Trait locus
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
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
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
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
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
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
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
IC = core.ReproductionWeight(name='Clocus inheritance', \
                             axes=['female_Clocus', 'male_Clocus', 'offspring_Clocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Clocus')])
IC.set( utils.nuclear_inheritance(n_alleles) )
IC_ = IC.extended()
print IC

#! Background locus D
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
ID = core.ReproductionWeight(name='Dlocus inheritance', \
                             axes=['female_Dlocus', 'male_Dlocus', 'offspring_Dlocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Dlocus')])
ID.set( utils.nuclear_inheritance(n_alleles) )
ID_ = ID.extended()
print ID

#! Background locus E
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
IE = core.ReproductionWeight(name='Elocus inheritance', \
                             axes=['female_Elocus', 'male_Elocus', 'offspring_Elocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Elocus')])
IE.set( utils.nuclear_inheritance(n_alleles) )
IE_ = IE.extended()
print IE

#! Background locus F
#!......................................................................
#$ ~    % we need this non-beaking space because the paragraph would otherwise be empty and the paragraph title would not be displayed
IF = core.ReproductionWeight(name='Flocus inheritance', \
                             axes=['female_Flocus', 'male_Flocus', 'offspring_Flocus'], \
                             config=config, \
                             unstack_levels=[2], \
                            )
n_alleles = len(ALLELES[LOCI.index('Flocus')])
IF.set( utils.nuclear_inheritance(n_alleles) )
IF_ = IF.extended()
print IF


#! Simulations runs
#!======================================================================
mode = None
#~ mode = 'report'      # create a report with pyreport

desc = """
- 2 populations linked by symmetrical migration
- populations 1 uninfected, pop. 2 Wolbachia-infected
- a different trait adaptive in each population (T1 in pop.1, T2 in pop.2)
- hybrid males are fully sterile due to divergence at 6 loci (A, B, C, D, E, and F):
  A1-B1, A1-D1, B1-C1, C1-D1, ... males are sterile
- simulation is over when the final equilibrium has been reached
- system is checked for stability of the Wolbachia infection pattern
"""
snum = 'mcrit_6_W'
rstore = storage.RunStore('/extra/flor/data/scenario_{0}.h5'.format(snum))
rnum = 1
try: rstore.select_scenario(snum, verbose=False)
except: rstore.create_scenario(snum, labels=(LOCI,ALLELES), description=desc)
try: rstore.remove_run(rnum, snum)
except: pass
migration_rate = m = 0.05           # symmetric migration
mig_decimals = 3
mig_prec = math.pow(10, -mig_decimals)
lower_mig = 0.0
upper_mig = 0.5
mig_diff = 1.
totalstarttime = time.time()
while mig_diff > mig_prec:
    #! Migration
    #!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print "RUN {0}".format(rnum)
    print "=================================="
    print
    print "MIGRATION:"
    parameters['m'] = m
    mig = np.array([[ 1-m,      m ], \
                    [   m,    1-m ]], float)
    M = core.MigrationWeight(name='migration', \
                             axes=['target', 'source'], \
                             config=config, \
                             arr=mig, \
                             m=m
                            )
    weights['migration'] = M.extended()
    print M


    # we can combine all reproduction weights that are not frequency-dependent:
    R_ = CI_ * F_ * T_ * \
         IT_ * IA_ * IB_ * IC_ * ID_ * IE_ * IF_ * \
         HMS_AB_ * \
         HMS_AC_ * HMS_BC_ * \
         HMS_AD_ * HMS_BD_ * HMS_CD_ * \
         HMS_AE_ * HMS_BE_ * HMS_CE_ * HMS_DE_ * \
         HMS_AF_ * HMS_BF_ * HMS_CF_ * HMS_DF_ * HMS_EF_
    weights['constant_reproduction'] = R_

    rstore.init_run(rnum, parameters, FSHAPE, init_len=100)

    if mode == 'report':
        progress = False
    else:
        progress = True

    n = 10000
    step = 10
    figs = []
    figsize = [20,11]

    #! Start
    #!----------------------------------------------------------------------
    print
    print "START:"
    print "----------------------------------"
    startfreqs = np.zeros(FSHAPE)
    startfreqs[0,1,0,1,0,1,0,0,0] = 1.   # pop1-A1-B0-C1-D0-E1-F0-T1-U
    startfreqs[1,0,1,0,1,0,1,1,1] = 1.   # pop2-A0-B1-C0-D1-E0-F1-T2-W
    starttime = time.time()
    metapop = core.MetaPopulation(startfreqs, config=config, generation=0, name='metapopulation')
    rstore.record_special_state(metapop.generation, 'start')
    rstore.dump_data(metapop)
    #! Plot
    #!......................................................................
    print
    print "PLOT:"
    fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
    figs.append(fig)
    show()
    #! Nucleocytotype frequencies
    #!......................................................................
    print "NUCLEOCYTOTYPE FREQUENCIES:"
    print metapop
    #! Locus overview
    #!......................................................................
    print "LOCUS OVERVIEW:"
    print metapop.overview(['Alocus', 'Blocus'], \
                           ['Clocus', 'Dlocus'], \
                           ['Elocus', 'Flocus'], \
                           'trait', 'cytotype')

    #! Final
    #!----------------------------------------------------------------------
    print "FINAL:"
    print "----------------------------------"
    progress = metapop.run(n, weights, threshold=threshold, step=step, runstore=rstore, progress=progress)
    #! Plot
    #!......................................................................
    print
    print "PLOT:"
    fig = rstore.plot_overview(generation=metapop.generation, figsize=figsize)
    figs.append(fig)
    show()
    #! Nucleocytotype frequencies
    #!......................................................................
    print "NUCLEOCYTOTYPE FREQUENCIES:"
    print metapop
    #! Locus overview
    #!......................................................................
    print "LOCUS OVERVIEW:"
    print metapop.overview(['Alocus', 'Blocus'], \
                           ['Clocus', 'Dlocus'], \
                           ['Elocus', 'Flocus'], \
                           'trait', 'cytotype')
    #! Critical migration rate
    #!......................................................................
    print "STABILITY of the HMS PATTERN:"
    if hms_stable(metapop, hms_prec):
        pos = 'below'
        lower_mig = m
    else:
        pos = 'above'
        upper_mig = m
    print "m = {0} is {1} the critical migration rate\n".format(m, pos)
    scenario, run = rstore.get_current()
    run.attrs['critical_migration_rate'] = pos
    #! Runtime
    #!......................................................................
    print "RUNTIME:"
    print 'Simulation run completed:'
    seconds = time.time()-starttime
    hhmmss = str(datetime.timedelta(seconds=int(seconds)))
    print 'Generation: {0} (Elapsed Time: {1})\n'.format(metapop.generation, hhmmss)
    

    #! Dynamics
    #!----------------------------------------------------------------------
    print "DYNAMICS:"
    print "----------------------------------"
    fig = rstore.plot_sums(figsize=figsize)
    figs.append(fig)
    show()

    rstore.flush()
    
    m = lower_mig + abs(upper_mig - lower_mig)/2.
    mig_diff = abs(upper_mig - lower_mig)
    rnum += 1

#! Runtime
#!======================================================================
seconds = time.time()-totalstarttime
hhmmss = str(datetime.timedelta(seconds=int(seconds)))
print '{0} simulation runs  (Elapsed Time: {1})\n'.format(rnum-1, hhmmss)

#! Critical migration rate
#!======================================================================
#! 
#! Lower bound approximation of the critical migration rate (HMS pattern is still stable at this migration rate)
#! 
#! Decimal precision:
print mig_decimals
#! 
#! Critical migration rate:
print lower_mig
    
rstore.close()
