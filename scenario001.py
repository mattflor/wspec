import sys, types, time, os, inspect, shutil, pprint, cPickle, gzip, tarfile, pprint
sys.path.append(".")             # pyreport needs this to know where to import modules from
import numpy as np
import numpy.random as npr
import pandas as pd
from pylab import show          # pyreport needs this to find figures

import core, storage, visualization
import utilities as utils


np.set_printoptions(precision=4, suppress=True, linewidth=200)
report = False                   # set this to True if running script through pyreport (vertical subplots)
scenarioname='scenario04_01'    #str(os.path.splitext(__file__)[0])


#~ class metapopulation(object):
    #~ def __init__(self, frequencies, config, generation=0, name='metapopulation', eq='undetermined'):
        #~ self.loci = config['LOCI']
        #~ self.n_loci = len(self.loci)
        #~ self.alleles = config['ALLELES']
        #~ self.repro_axes = config['REPRO_AXES']  # reproduction_axes(loci)
        #~ self.repro_dim = config['REPRO_DIM']    #len(self.repro_axes)
        #~ assert np.shape(frequencies) == utils.list_shape(self.alleles)
        #~ self.freqs = frequencies
        #~ self.ndim = self.freqs.ndim
        #~ self.shape = self.freqs.shape
        #~ self.size = self.freqs.size
        #~ self.normalize()
        #~ self._allele_idxs = utils.make_allele_dictionary(self.loci, self.alleles)
        #~ self.labels = utils.panda_index(self.alleles, self.loci)
        #~ self.populations = self.alleles[0]
        #~ self.n_pops = len(self.populations)
        #~ self.generation = generation
        #~ self.name = name
        #~ self.eq = eq
        #~ self.panda = pd.Series(self.freqs.flatten(), index=self.labels, name=name)
        #~ self.livechart = False
        #~ 
        #~ self.male_axes = utils.reproduction_axes(self.loci, 'male')
        #~ self.male_idxs = [self.repro_axes.index(a) for a in self.male_axes]
        #~ self.female_axes = utils.reproduction_axes(self.loci, 'female')
        #~ self.female_idxs = [self.repro_axes.index(a) for a in self.female_axes]
        #~ self.offspring_axes = utils.reproduction_axes(self.loci, 'offspring')
        #~ self.offspring_idxs = [self.repro_axes.index(a) for a in self.offspring_axes]
    #~ 
    #~ def __str__(self):
        #~ if not self.isuptodate():
            #~ self.update()
        #~ s = "{0}\nName: {1}\nGeneration: {2}\nEQ: {3}".format( \
                #~ self.panda.unstack([0,-1]).to_string(float_format=myfloat), \
                #~ self.name, \
                #~ self.generation, \
                #~ self.eq )
        #~ return s
    #~ 
    #~ def overview(self):
        #~ s = str(self.get_sums_pd([1,2]).unstack(2)) + '\n'
        #~ s += 'Name: background loci\n\n'
        #~ for loc in self.loci[3:]:
            #~ s += str(self.get_sums_pd(loc).unstack(1)) + '\n'
            #~ s += 'Name: {0}\n\n'.format(loc)
        #~ return s
    #~ 
    #~ def normalize(self):
        #~ s = sum_along_axes(self.freqs, 0)          # first axis are `populations`
        #~ self.freqs /= extend(s, self.ndim, 0)      # in-place, no copy
    #~ 
    #~ def isuptodate(self):
        #~ return np.all(self.panda.values == self.freqs.flatten())
    #~ 
    #~ def update(self):
        #~ self.panda.data = self.freqs.flatten()
    #~ 
    #~ def store_freqs(self, filename='freqs.npy'):
        #~ np.save(filename, self.freqs)
    #~ 
    #~ def load_freqs(self, filename='freqs.npy'):
        #~ freqs = np.load(filename)
        #~ assert np.shape(freqs) == self.shape
        #~ self.freqs = freqs
        #~ self.update()
    #~ 
    #~ def load(self, frequencies, generation):
        #~ self.generation = generation
        #~ self.freqs = frequencies[str(generation)]
    #~ 
    #~ def get_sum(self, allele, pop):
        #~ if not isinstance(pop,int): pop = self.populations.index(pop)
        #~ l,a = self._allele_idxs[allele]
        #~ return sum_along_axes(self.freqs, [0,l])[pop,a]
#~ 
    #~ def get_sums(self, locus, pop=None):
        #~ level = [0]
        #~ if not isinstance(locus, list):
            #~ locus = [locus]
        #~ for loc in locus:
            #~ if isinstance(loc, int): level.append(loc)
            #~ else: level.append( self.loci.index(loc) )
        #~ if pop or pop==0:
            #~ if not isinstance(pop,int):
                #~ popname, pop = pop, self.populations.index(pop)
            #~ else:
                #~ popname = self.populations[pop]
            #~ return sum_along_axes(self.freqs, level)[pop]
        #~ return sum_along_axes(self.freqs, level)
    #~ 
    #~ def get_sums_pd(self, locus, pop=None):
        #~ if not self.isuptodate():
            #~ self.update()
        #~ level = [0]
        #~ if not isinstance(locus, list):
            #~ locus = [locus]
        #~ for loc in locus:
            #~ if isinstance(loc, int): level.append(loc)
            #~ else: level.append( self.loci.index(loc) )
        #~ p = self.panda.sum(level=level)
        #~ if pop or pop==0:
            #~ if isinstance(pop,int):
                #~ pop = self.populations[pop]
            #~ return p[pop]
        #~ return p
    #~ 
    #~ def introduce_allele(self, pop, allele, intro_freq, advance_generation_count=True):
        #~ """
        #~ `pop` : population index or name
        #~ `allele` : allele name
        #~ `intro_freq` : introduction frequency of `allele`
        #~ """
        #~ if not isinstance(pop,int):
            #~ pop = self.populations.index(pop)
        #~ loc,al = self._allele_idxs[allele]
        #~ lfreqs = sum_along_axes(self.freqs, [0,loc])[pop]
        #~ try:
            #~ assert lfreqs[al] == 0.
        #~ except AssertionError:
            #~ raise AssertionError, 'allele `{0}` already present in {1}'.format(allele,self.populations[pop])
        #~ locus_sums = np.sum( self.freqs, axis=loc )[pop]   # freqs: (2,2,3,2) --> (2,2,2)[pop] --> (2,2)
        #~ idxs = [slice(None,None,None) for i in range(self.ndim)]
        #~ idxs[0] = pop
        #~ idxs[loc] = al
        #~ self.freqs[pop] *= 1 - intro_freq
        #~ self.freqs[idxs] = intro_freq * locus_sums
        #~ if advance_generation_count:
            #~ self.generation += 1
        #~ self.eq = 'not determined'
    #~ 
    #~ def run(self, n=1000, step=100, threshold=1e-4, chart=None):
        #~ """
        #~ Simulate next `n` generations. Abort if difference between consecutive
        #~ generations is smaller than `threshold`.
        #~ `step` is used for plotting only.
        #~ 
        #~ To enable live stripcharting, pass a stripchart instance to `chart`.
        #~ """
        #~ global SR, TP
        #~ self.chart = chart
        #~ n += self.generation
        #~ thresh = threshold/self.size   # on average, each of the frequencies should change less than `thresh`
        #~ pt = SR.pt
        #~ species_preferences = [(sno,prefs) for pname,sno,prefs in SR.preferences]
        #~ trait_preferences = [(pno,prefs) for pname,pno,prefs in TP.preferences]
        #~ still_changing = True
        #~ while still_changing and self.generation < n:
            #~ previous = np.copy(self.freqs)
            #~ ### migration ##################################
            #~ self.freqs = np.sum(self.freqs[np.newaxis,...] * M_, 1)   # sum over source axis
            #~ self.normalize()
            #~ 
            #~ ### viability selection ########################
            #~ self.freqs = self.freqs * V_
            #~ self.normalize()
            #~ 
            #~ ### reproduction ###############################
            #~ # species recognition:
            #~ SR.set_to_ones()
            #~ ABfreqs = self.get_sums(['backA','backB'])
            #~ for sno, prefs in species_preferences:
                #~ for pop, ano, bno, pr in prefs:
                    #~ R = 1./(1-pr*pt*(1-ABfreqs[pop,ano,bno]))
                    #~ SR.array[pop,sno] *= (1-pr)*R
                    #~ SR.array[pop,sno,ano,bno] = R
            #~ SR.array = np.nan_to_num(SR.array)
            #~ SR_ = SR.extended()
            #~ 
            #~ # trait preferences:
            #~ TP.set_to_ones()
            #~ traitfreqs = self.get_sums('trait')
            #~ for pno, prefs in trait_preferences:
                #~ for pop, tno, pr in prefs:
                    #~ R = 1./(1-pr*pt*(1-traitfreqs[pop,tno]))
                    #~ TP.array[pop,pno] *= (1-pr)*R
                    #~ TP.array[pop,pno,tno] = R
            #~ TP.array = np.nan_to_num(TP.array)           # replace NaN with zero (happens when pr=pt=1 and x=0)
            #~ TP_ = TP.extended()
            #~ 
            #~ 
            #~ # offspring production:
            #~ females = extend( self.freqs, REPRO_DIM, self.female_idxs )
            #~ males = extend( self.freqs, REPRO_DIM, self.male_idxs )
            #~ offspring = sum_along_axes( females * males * R_ * SR_ * TP_, self.offspring_idxs )
            #~ self.freqs = offspring
            #~ self.normalize()
            #~ 
            #~ if self.generation % step == 0:
                #~ GENS.append(self.generation)
                #~ FREQS.append(self.freqs)
                #~ allele_freqs = []
                #~ for i,pop in enumerate(self.populations):
                    #~ allele_freqs.append([])
                    #~ for al in chartlabels[pop]:
                        #~ allele_freqs[i].append( self.get_sum(al, pop) )
                #~ update_plot_data(self.generation, allele_freqs)
                #~ if self.chart:
                    #~ self.chart.update()
            #~ 
            #~ self.generation += 1
            #~ still_changing = diff(self.freqs, previous) > thresh
  #~ 
        #~ self.eq = not still_changing
        #~ if self.chart:
            #~ self.chart.finalize()

def update_plot_data(x, y):
    global XDATA, YDATA
    XDATA.append(x)
    for pop,row in enumerate(y):
        for al,value in enumerate(row):
            YDATA[pop][al].append(value)


def store_sim(filename=scenarioname, cleanup=False):
    path = filename        # use the same for both
    if not os.path.isdir(path):
        os.mkdir(path)
    simname = os.path.join(path,'simulation.gz')
    chartname = os.path.join(path,'chart.pdf')
    df = gzip.open(simname, 'wb')
    for d in [config, parameters, weights, frequencies]:
        cPickle.dump(d, df)
    df.close()
    chart.fig.savefig(chartname)
    tar = tarfile.open(filename+'.tar', 'w')
    tar.add(simname)
    tar.add(chartname)
    tar.close()
    if cleanup:
        # remove temporary files:
        for name in [simname, chartname]:
            os.remove(name)
        os.rmdir(path)

def load_sim(filename):
    """
    Load simulation data for further interactive work.    
    """
    tar = tarfile.open(filename+'.tar', 'r')
    tar.extractall()
    tar.close()
    simfile = os.path.join(filename, 'simulation.gz')
    df = gzip.open(simfile, 'rb')
    config = cPickle.load(df)
    parameters = cPickle.load(df)
    weights = cPickle.load(df)
    frequencies = cPickle.load(df)
    df.close()
    return dict(config=config, parameters=parameters, weights=weights, frequencies=frequencies)

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
           ['T1','T2','T3','T4'], \
           ['P0', 'P1', 'P2'], \
           ['U', 'W']
          ]
loc_width = len(max(LOCI, key=len))
for i,loc in enumerate(LOCI):
    print "%-*s  :\t%s" % (loc_width, loc, ', '.join(ALLELES[i]))
    
#! Parameters
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
migration_rate = m = 0.02           # symmetric migration!
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
parameters = dict(m=m, s=s, lCI=l, f=f, t=t, pt=pt, pr_s1=pr_s1, pr_s2=pr_s2, pr_t1=pr_t1, pr_t2=pr_t2, intro=intro)           # dictionary for storing simulation
par_width = len(max(parameters.keys(), key=len))
for p,v in sorted(parameters.items()):
    print "%-*s  :\t%s" % (par_width, p, v)
print


# setting up some scenario wide stuff
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

# Set up plot
# `chartlabels` define what is being plotted             
chartlabels = {'pop1': ['A1', 'S1', 'P0', 'P1', 'T1', 'W'], \
               'pop2': ['A1', 'S1', 'P0', 'P1', 'T2', 'W'], \
               'pop3': ['A1', 'S1', 'P0', 'P1', 'T3', 'W'], \
               'pop4': ['A2', 'S2', 'P0', 'P2', 'T4', 'W']
              }
XDATA = []
YDATA = [[] for pop in range(N_POPS)]
for i,pop in enumerate(POPULATIONS):
    for al in chartlabels[pop]:
        YDATA[i].append( [] )
GENS = []
FREQS = []
        
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
viab = np.array([[1+s,      1,      1,      1], \
                 [  1,    1+s,      1,      1], \
                 [  1,      1,    1+s,      1], \
                 [  1,      1,      1,    1+s]], float)
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
TI.set( utils.nuclear_inheritance(4) )
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

#~ random_frequencies = npr.random(FSHAPE)
#~ metapop = metapopulation(random_frequencies, config=config, generation=0, name='metapopulation')
#~ metapop.normalize()
#~ 
#~ if report == True:
    #~ chart = None
#~ else:
    #~ chart = stripchart(chartlabels)
    #~ plt.show()
    #~ 
    
#! Simulation
#!======================================================================
n = 50
frequencies = {}           # dictionary for storing simulation

#! Start frequencies
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
startfreqs = np.zeros(FSHAPE)
startfreqs[0,0,0,0,0,0,0] = 1.                   # pop1-A1-B1-S1-T1-P0-U
startfreqs[1,0,0,0,1,0,0] = 1.                   # pop2-A1-B1-S1-T2-P0-U
startfreqs[2,0,0,0,2,0,0] = 1.                   # pop3-A1-B1-S1-T3-P0-U
startfreqs[3,1,1,1,3,0,1] = 1.                   # pop4-A2-B2-S2-T4-P0-W
metapop = core.MetaPopulation(startfreqs, config=config, generation=0, name='metapopulation')
print metapop
frequencies['0'] = startfreqs

#! Migration-selection equilibrium
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.run(n, weights, threshold=1e-4, step=10, runstore=r)
print metapop
frequencies[str(metapop.generation)] = metapop.freqs

#! Introduction of preference allele P1
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.introduce_allele('pop3', 'P1', intro_freq=intro, advance_generation_count=True)
#~ metapop.introduce_allele('pop4', 'P2', intro_freq=intro)
print metapop
frequencies[str(metapop.generation)] = metapop.freqs
#~ GENS.append(metapop.generation)
#~ FREQS.append(metapop.freqs)

#! Equilibrium
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.run(n, weights, threshold=1e-4, step=10, runstore=r)
print metapop
frequencies[str(metapop.generation)] = metapop.freqs
#~ GENS.append(metapop.generation)
#~ FREQS.append(metapop.freqs)

#! Introduction of preference allele P2
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.introduce_allele('pop4', 'P2', intro_freq=intro, advance_generation_count=True)
#~ metapop.introduce_allele('pop4', 'P2', intro_freq=intro)
print metapop
frequencies[str(metapop.generation)] = metapop.freqs
#~ GENS.append(metapop.generation)
#~ FREQS.append(metapop.freqs)

#! Final state
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
metapop.run(n, weights, threshold=1e-4, step=10, runstore=r)
print metapop
frequencies[str(metapop.generation)] = metapop.freqs
#~ GENS.append(metapop.generation)
#~ FREQS.append(metapop.freqs)

#! Loci (sums)
#!----------------------------------------------------------------------
print metapop.overview()

#~ if report:
    #~ chart = stripchart(chartlabels)
    #~ chart.draw()
    #~ chart.finalize()
    
#! Dynamic weights (final states)
#!======================================================================
#! Sexual selection (final)
#!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#! Species recognition (final)
#!----------------------------------------------------------------------
print SR

#! Trait preference (final)
#!----------------------------------------------------------------------
print TP

#~ #! Chart
#~ #!======================================================================
#~ show()
#~ 
#~ store_sim()
