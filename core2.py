"""
.. module:: core
   :platform: Unix
   :synopsis: Core functionality and classes

.. moduleauthor:: Matthias Flor <matthias.c.flor@gmail.com>

"""
import sys
import numpy as np
default_dtype = np.float32

import pandas as pd
import utilities as utils
import progressbar as pbar
extend = utils.extend
myfloat = utils.myfloat
sum_along_axes = utils.sum_along_axes
import pdb
reload(pbar)

class Weight(object):
    """
    Weight base class.
    
    Not usable on its own because no panda respresentation is created.
    """
    def __init__(self, name, axes, arr=None, unstack_levels=-1, **parameters):
        self.name = name
        self.axes = axes
        self.array = np.array(arr, dtype=default_dtype)
        self.unstack_levels = unstack_levels
        self.parameters = parameters
        self.__dict__.update(parameters)
    
    def make_panda(self, labels):
        pd_idx = utils.panda_index(labels, self.axes)
        self.panda = pd.Series(self.array.flatten(), index=pd_idx, name=self.name)
    
    def configure_extension(self, dim, pos):
        self.extdim = dim
        self.extpos = pos
        
    def set(self, arr):
        """
        Set weight array to `arr` and update panda representation.
        """
        assert np.shape(arr) == self.shape
        self.array = np.array(arr, dtype=default_dtype)
        self.panda.data = self.array
    
    def set_to_ones(self):
        self.array = np.ones(self.shape, dtype=default_dtype)
    
    def isuptodate(self):
        """
        Return True if panda representation is up to date.
        """
        return np.all( self.array.flatten() == self.panda.values )
        
    def update(self):
        """
        Update panda representation.
        """
        self.panda.data = self.array
    
    def extended(self):
        return extend(self.array, self.extdim, self.extpos)
        
    def __str__(self):
        """
        Nicely formatted string output of the reproduction weight. We 
        just use the panda Series output.
        """
        if not self.isuptodate():
            self.update()
        if self.unstack_levels:
            s = '{0}\nName: {1}\n'.format( self.panda.unstack(self.unstack_levels), self.name )
        else:
            s = str(self.panda) + '\n'
        pars = ''
        for k,v in sorted(self.parameters.items()):
          pars += '{0}: {1}\n'.format(k,v)
        return s + pars.rstrip()
    
    def str_myfloat(self):
        if not self.isuptodate():
            self.update()
        if self.unstack_levels:
            s = '{0}\nName: {1}\n'.format( self.panda.unstack(self.unstack_levels).to_string(float_format=myfloat), self.name )
        else:
            s = '{0}\n'.format( self.panda.to_string(float_format=myfloat) )
        pars = ''
        for k,v in sorted(self.parameters.items()):
          pars += '{0}: {1}\n'.format(k,v)
        return s + pars.rstrip()
    

class MigrationWeight(Weight):
    def __init__(self, \
                 name='migration', \
                 axes=['target', 'source'], \
                 config=None, \
                 arr=None, \
                 unstack_levels=-1, \
                 **parameters):
        Weight.__init__(self, name, axes, arr, unstack_levels, **parameters)
        labels = utils.get_alleles(['population','population'], config=config)
        self.shape = utils.list_shape(labels)
        if arr == None:
            arr = np.zeros( self.shape, dtype=default_dtype )
        self.array = np.array(arr, dtype=default_dtype)
        self.make_panda(labels)
        self.configure_extension( dim=1+config['N_LOCI'], pos=[0,1] )

class ViabilityWeight(Weight):
    def __init__(self, \
                 name='viability selection', \
                 axes=['population', 'trait'], \
                 config=None, \
                 arr=None, \
                 unstack_levels=-1, \
                 **parameters):
        Weight.__init__(self, name, axes, arr, unstack_levels, **parameters)
        labels = utils.get_alleles(axes, config=config)
        self.shape = utils.list_shape(labels)
        if arr == None:
            arr = np.zeros( self.shape, dtype=default_dtype )
        self.array = np.array(arr, dtype=default_dtype)
        self.make_panda(labels)
        pos = [config['LOCI'].index(a) for a in axes]
        self.configure_extension( dim=config['N_LOCI'], pos=pos )

class ReproductionWeight(Weight):
    """
    Weights to be used in the reproduction step of next generation
    production.
    
    Mainly, this class takes care of
    - printing a nice panda version of the weight array and
    - enabling autmatic extension to the correct shape needed in the 
      reproduction step.
    
    Constant weights can be instantiated directly as instances of this
    class whereas dynamic weights should be defined in the scenario 
    files as custom classes inheriting from this class.
    The class method `dynamic` should be used to achieve a dynamic 
    update of the weight array.
    """
    def __init__(self, name, axes, config, arr=None, unstack_levels=[], **parameters):
        """
        Args:
            name: str
                name of the weight
            axes: list of strings
                list of axes names
            config: dict
                scenario configuration
            arr: ndarray
                if `arr` is not provided on initialization, it will be
                populated with zeros and **must** be set afterwords with
                the `set` method
            unstack_levels: int, string, or list of these
                level(s) to unstack panda Series
            parameters: dict
                dictionary of parameter names (keys) and values (values)
        """
        Weight.__init__(self, name=name, axes=axes, arr=arr, unstack_levels=unstack_levels, **parameters)
        labels = utils.make_reproduction_allele_names(axes, config)
        self.shape = utils.list_shape( labels )
        if arr == None:
            arr = np.zeros( self.shape, float )
        self.array = np.array(arr, dtype=default_dtype)
        self.make_panda(labels)
        dim = config['REPRO_DIM']
        repro_axes = config['REPRO_AXES']
        pos = [repro_axes.index(ax) for ax in axes]
        self.configure_extension( dim=dim, pos=pos )

class PreferenceWeight(ReproductionWeight):
    def __init__(self, name, axes, pref_desc, config, unstack_levels=[], **parameters):
        """
        Args:
            name, axes, config, unstack_levels, and parameters: see parent class
            pref_desc: dict describing preferences
                e.g.: {'S1': {'pop1': ('A1-B1', 0.9), \
                              'pop2': ('A1-B1', 0.9)}, 
                       'S2': {'pop1': ('A2-B2', 0.9), \
                              'pop2': ('A2-B2', 0.9)}}
                This description will be translated into a list that is
                easier to use in indexing the array.
        """
        ReproductionWeight.__init__(self, name=name, axes=axes, config=config, unstack_levels=unstack_levels, **parameters)
        # use config for determining preference allele indexes
        self.pref_desc = pref_desc
        preferences = []
        for pref_allele,pop_prefs in sorted(pref_desc.items()):
            prefidx = config['ADICT'][pref_allele][1]   # retrieve allele index
            for pop,(cues,pr) in sorted(pop_prefs.items()):
                if pop == 'all pops':   # same preference in all populations
                    popidx = slice(None,None,None)
                else:
                    popidx = config['ADICT'][pop][1]
                cues = cues.split('-')
                cueidx = tuple( [config['ADICT'][c][1] for c in cues] )   # get cue allele indexes
                preferences.append( ((popidx,prefidx)+cueidx, pr) ) # tuple of all indexes together (as a tuple) and the rejection probability
        self.preferences = preferences
    
    def calculate(self, x):
        """
        Args:
            x: ndarray
                frequency array of preferred traits in the appropriate shape
            pt: float
                transition probability
        """
        self.set_to_ones()
        for idx,pr in self.preferences:         # idx: complete indexes
            #~ pdb.set_trace()
            idx2 = idx[:1] + idx[2:]          # idx2: preference allele index removed
            tmp = 1./(1-pr*self.pt*(1-x[idx2]))
            tmp_ = extend(tmp, dim=len(idx2), pos=0)  #  tmp[:,np.newaxis,np.newaxis] 
            self.array[idx[:2]] *= (1-pr)*tmp_   # idx[:2]: preferred trait indexes removed
            self.array[idx] = tmp
        self.array = np.nan_to_num(self.array)
        
    #~ def calculate(self, preferred):
        #~ self.set_to_ones()
        #~ for sno, prefs in self.preferences:
            #~ for pop, ano, bno, pr in prefs:
                #~ tmp = 1./(1-pr*pt*(1-preferred[pop,ano,bno]))
                #~ self.array[pop,sno] *= (1-pr)*tmp
                #~ self.array[pop,sno,ano,bno] = tmp
        #~ self.array = np.nan_to_num(self.array)

class GeneralizedPreferenceWeight(ReproductionWeight):
    def __init__(self, name, axes, pref_desc, config, unstack_levels=[], **parameters):
        """
        Args:
            name, axes, config, unstack_levels, and parameters: see parent class
            pref_desc: dict describing preferences
                e.g.: {'P0': {'baseline': 0.},              # 0. is the default baseline!
                       'P1': {'baseline': 0.9, 'T3': 0.},   # all traits not explicitely mentioned will be rejected with the baseline probability
                       'P2': {'baseline': 0.8, 'T4': 0.}
                      }
                This description will be translated into an array containing
                the rejection probabilities that can be accessed by the 
                preference allele index and cue indexes.
        """
        ReproductionWeight.__init__(self, name=name, axes=axes, config=config, unstack_levels=unstack_levels, **parameters)
        self.cue_axes = []
        split_axes = [a.split('_') for a in axes]
        for a in split_axes:
            if a[0]=='female':
                self.pref_locus = a[1]
            elif a[0]=='male':
                self.cue_axes.append(a[1])
        fshape = config['FSHAPE']
        loci = config['LOCI']
        alleles = config['ALLELES']
        adict = config['ADICT']
        n_prefs = fshape[loci.index(self.pref_locus)]
        self.cshape = tuple( [fshape[loci.index(a)] for a in self.cue_axes] )  # cue_shape
        rprobs = np.zeros( (n_prefs,)+self.cshape, float )   # rejection probabilities array with default baseline of 0.
        self.pref_desc = pref_desc
        for pref_allele,prefs in sorted(pref_desc.items()):
            prefidx = adict[pref_allele][1]   # retrieve allele index
            keys = sorted(prefs.keys())
            if 'baseline' in keys:
                pr = prefs['baseline']
                rprobs[prefidx] = pr
            for cue,pr in sorted(prefs.items()):
                if cue == 'baseline':
                    break
                cues = cue.split('-')
                cueidx = tuple( [adict[c][1] for c in cues] )   # get cue allele indexes
                rprobs[(prefidx,)+cueidx] = pr
                #~ preferences.append( ((popidx,prefidx)+cueidx, pr) ) # tuple of all indexes together (as a tuple) and the rejection probability
        self.rejection_probabilities = self.rprobs = rprobs   # shape: (pref, cue1, cue2, ...)
        names = [self.pref_locus] + self.cue_axes
        labels = []
        for a in names:
            labels.append(alleles[loci.index(a)])
        idx = utils.panda_index(labels, names)
        self.rpanda = pd.Series(rprobs.flatten(), index=idx, name='rejection probabilities')
        
    
    def calculate(self, x):
        """
        Args:
            x: ndarray
                frequency array of preferred traits in the appropriate shape
            pt: float
                transition probability
        """
        #~ self.set_to_ones()
        rej = self.rprobs[np.newaxis,...]    # bring rejection probabilities to correct shape
        cues = x[:,np.newaxis,...]           # same for preference cues
        norm = 1. - self.pt * utils.sum_along_axes(rej*cues, [0,1])  # sum along population and preference a.k.a. sum over all cues
        self.array = np.nan_to_num( (1.-rej)/norm[...,np.newaxis] )
    
    def __str__(self):
        s = Weight.__str__(self) + '\n'
        ndim = self.rprobs.ndim
        s += '{0}:\n{1}\n'.format( self.rpanda.name, self.rpanda.unstack([1]*(ndim-1)) )
        #~ s += 'rejection probabilities:\n'
        #~ for pref,vals in sorted(self.pref_desc.items()):
            #~ s += '    {0}    '.format(pref)
            #~ for cue,pr in vals:
                #~ s += '{0}:    {1}\n'.format(p, str(v).translate(None, "'{}"))
        return s
    
    
class DummyProgressBar(object):
    def update(self, val):
        pass
    def finish(self):
        pass
    
def generate_progressbar(progress_type='real'):
    if progress_type == 'real':
        widgets=['Generation: ', pbar.Counter(), ' (', pbar.Timer(), ')']
        return pbar.ProgressBar(widgets=widgets, maxval=1e6).start()   # don't provide maxval!
    elif progress_type == 'dummy':
        return DummyProgressBar()
    else:
        raise TypeError, "progress_type `{0}` unknown; use `real` or `dummy` instead".format(progress_type)

class MetaPopulation(object):
    def __init__(self, frequencies, config, generation=0, name='metapopulation', eq='undetermined'):
        self.loci = config['LOCI']
        self.n_loci = len(self.loci)
        self.alleles = config['ALLELES']
        #~ self.repro_axes = config['REPRO_AXES']  # reproduction_axes(loci)
        #~ self.repro_dim = config['REPRO_DIM']    #len(self.repro_axes)
        assert np.shape(frequencies) == utils.list_shape(self.alleles)
        self.freqs = frequencies
        self.ndim = self.freqs.ndim
        self.shape = self.freqs.shape
        self.size = self.freqs.size
        self.normalize()
        self.allele_idxs = config['ADICT']
        self.populations = self.alleles[0]
        self.n_pops = len(self.populations)
        self.generation = generation
        self.name = name
        self.eq = eq
        labels = utils.panda_index(self.alleles, self.loci)
        self.panda = pd.Series(self.freqs.flatten(), index=labels, name=name)
        
        r_axes = config['REPRO_AXES']
        self.repro_axes = {'all': r_axes}
        #~ self.repro_shape = 3 * self.shape
        self.repro_dim = config['REPRO_DIM']
        self.repro_idxs = {}
        for who in ['female', 'male', 'offspring']:
            w_axes = utils.reproduction_axes(self.loci, who)
            self.repro_axes[who] = w_axes
            self.repro_idxs[who] = [r_axes.index(a) for a in w_axes]
                
    def __str__(self):
        """
        Returns nicely formatted string representation of metapopulation 
        as unstacked panda series.
        """
        if not self.isuptodate():
            self.update()
        s = "{0}\nName: {1}\nGeneration: {2}\nEQ: {3}\n".format( \
                self.panda.unstack([0,-1]).to_string(float_format=utils.myfloat), \
                self.name, \
                self.generation, \
                self.eq )
        return s
    
    def overview(self, *args):
        """
        Return nicely formatted string representation of locus sums.
        
        If arguments are passed then each argument must be a locus name
        or a list of locus names.
        """
        s = ''
        if not args:
        
            args = self.loci[1:]
        for a in args:
            if isinstance(a, list):
                s += str(self.get_sums_pd(a).unstack([-2,-1])) + '\n'
                s += 'Name: {0}\n\n'.format(', '.join(a))
            else:
                s += str(self.get_sums_pd(a).unstack(1)) + '\n'
                s += 'Name: {0}\n\n'.format(a)
        return s
        #~ s = str(self.get_sums_pd([1,2]).unstack(2)) + '\n'
        #~ s += 'Name: background loci\n\n'
        #~ for loc in self.loci[3:]:
            #~ s += str(self.get_sums_pd(loc).unstack(1)) + '\n'
            #~ s += 'Name: {0}\n\n'.format(loc)
        #~ return s
    
    def normalize(self):
        """
        Normalize frequencies so that they sum up to one in each 
        population.
        """
        s = sum_along_axes(self.freqs, 0)          # first axis are `populations`
        self.freqs /= extend(s, self.ndim, 0)      # in-place, no copy
    
    def isuptodate(self):
        """
        Return True if panda representation is up to date.
        """
        return np.all(self.panda.values == self.freqs.flatten())
    
    def update(self):
        """
        Update panda representation.
        """
        self.panda.data = self.freqs.flatten()
    
    def load_freqs_from_file(self, g, filename, snum, rnum):
        self.freqs = storage.get_frequencies(g, filename, snum, rnum)
        
    def get_sum(self, allele, pop):
        """
        Return the summed frequency of `allele` in `pop`.
        
        Args:
            allele: string
                allele name
            pop: int or string
                population index or name
        
        Returns:
            out: float
        """
        if not isinstance(pop,int): pop = self.allele_idxs(pop)[1]
        l,a = self.allele_idxs[allele]
        return sum_along_axes(self.freqs, [0,l])[pop,a]

    def get_sums(self, locus, pop=None):
        """
        Return the summed frequency at `locus` (in `pop` if given, or
        in all populations).
        
        Args:
            locus: int or string or list of these
                locus indexes or names
            pop: int or string
                population index or name
        
        Returns:
            out: ndarray
        """
        level = [0]
        if not isinstance(locus, list):
            locus = [locus]
        for loc in locus:
            if isinstance(loc, int): level.append(loc)
            else: level.append( self.loci.index(loc) )
        if pop or pop==0:
            if not isinstance(pop,int):
                popname, pop = pop, self.allele_idxs(pop)[1]
            else:
                popname = self.populations[pop]
            return sum_along_axes(self.freqs, level)[pop]
        return sum_along_axes(self.freqs, level)
    
    def all_sums(self):
        """
        Returns:
            out: list of ndarrays
                list of loci sums (each locus sum is an ndarray)
        """
        sums = []
        for locus in self.loci[1:]:
            sums.append( self.get_sums(locus) )
        return sums
    
    def get_sums_pd(self, locus, pop=None):
        """
        Return the summed frequency at `locus` (in `pop` if given, or
        in all populations) as a panda series for nice print output.
        
        Args:
            locus: int or string or list of these
                locus indexes or names
            pop: int or string
                population index or name
        
        Returns:
            out: ndarray
        """
        if not self.isuptodate():
            self.update()
        level = [0]
        if not isinstance(locus, list):
            locus = [locus]
        for loc in locus:
            if isinstance(loc, int): level.append(loc)
            else: level.append( self.loci.index(loc) )
        p = self.panda.sum(level=level)
        if pop or pop==0:
            if isinstance(pop,int):
                pop = self.populations[pop]
            return p[pop]
        return p
    
    def introduce_allele(self, pop, allele, intro_freq, advance_generation_count=True):
        """
        Introduce `allele` into `pop` with frequency `intro_freq`.
        
        The introduction is a way such that the summed frequencies of
        at all other loci are unaffected by the new allele. The allele 
        must not be present in the population already.
        If advance_generation_count is True, the generation of the 
        metapopulation is advanced by one.
        
        Args:
            pop: int or string
                population index or name
            allele: string
                allele name
            intro_freq: float in interval [0, 1]
                introduction frequency of `allele`
        """
        if not isinstance(pop,int):
            pop = self.allele_idxs[pop][1]
        loc,al = self.allele_idxs[allele]
        lfreqs = sum_along_axes(self.freqs, [0,loc])[pop]
        try:
            assert lfreqs[al] == 0.
        except AssertionError:
            raise AssertionError, 'allele `{0}` already present in {1}'.format(allele,self.populations[pop])
        locus_sums = np.sum( self.freqs, axis=loc )[pop]   # freqs: (2,2,3,2) --> (2,2,2)[pop] --> (2,2)
        idxs = [slice(None,None,None) for i in range(self.ndim)]
        idxs[0] = pop
        idxs[loc] = al
        self.freqs[pop] *= 1 - intro_freq
        self.freqs[idxs] = intro_freq * locus_sums
        if advance_generation_count:
            self.generation += 1
        self.eq = 'not determined'
    
    def run(self, n, weights, step=100, threshold=1e-4, runstore=None, progress=True):
        """
        Simulate next `n` generations. Abort if average overall difference 
        between consecutive generations is smaller than `threshold` (i.e.
        an equilibrium has been reached).
                
        Args:
            weights: dictionary of weights to be used in the calculation
                     of the next generation frequencies
            n: int
                maximum number of generations to run
            step: int
                frequencies are stored every `step` generations
            threshold: float
                the `threshold` is divided by the frequency size 
                (arr.ndim) to calculate `thresh`, and the simulation run
                is stopped if the average difference between consecutive
                generations has become smaller than `thresh`.
            runstore: storage.runstore instance
                if provided, simulation run is stored in datafile
            progress: progressbar.ProgressBar instance
                if none is provided, a new one is created
        """
        MIG = weights['migration']
        VIAB_SEL = weights['viability_selection']
        REPRO_CONST = weights['constant_reproduction']
        dyn_repro_weights = weights['dynamic_reproduction']
        pt = dyn_repro_weights[0][0].pt
        #~ SR,TP = weights['dynamic_reproduction']
        #~ pt = SR.pt
        
        self.runstore = runstore
        n += self.generation
        thresh = threshold/self.size   # on average, each of the frequencies should change less than `thresh` if an equilibrium has been reached
        
        still_changing = True
        if isinstance(progress, (pbar.ProgressBar, DummyProgressBar)):
            pass   # reuse progressbar
        elif progress is False:
            progress = generate_progressbar('dummy')
        elif progress is True:
            progress = generate_progressbar('real')
            #~ widgets=['Generation: ', pbar.Counter(), ' (', pbar.Timer(), ')']
            #~ progress = pbar.ProgressBar(widgets=widgets, maxval=1e6).start()   # don't provide maxval! , fd=sys.stdout
        else:
            raise TypeError, "`progress` must be True, False, or an existing progressbar instance"
        while still_changing and self.generation < n:
            # data storage:
            if self.runstore != None:
                if self.generation % step == 0:
                    #~ self.runstore.dump_data(self.generation, self.freqs, self.all_sums())
                    self.runstore.dump_data(self)
                    #~ self.runstore.flush()
                    
            previous = np.copy(self.freqs)
            
            ### migration ##################################
            self.freqs = np.sum(self.freqs[np.newaxis,...] * MIG, 1)   # sum over `source` axis
            self.normalize()
            
            ### viability selection ########################
            self.freqs = self.freqs * VIAB_SEL
            self.normalize()
            
            ### reproduction ###############################
            #~ # species recognition:
            #~ SR.calculate( self.get_sums(['backA','backB']) )
            #~ 
            #~ # trait preferences:
            #~ TP.calculate( self.get_sums('trait') )
            REPRO_DYN = 1. #np.ones( (1,)*self.repro_dim )
            for DRW, target_loci in dyn_repro_weights:
                DRW.calculate( self.get_sums(target_loci) )
                REPRO_DYN = REPRO_DYN * DRW.extended()
            
            # offspring production:
            females = extend( self.freqs, self.repro_dim, self.repro_idxs['female'] )
            males = extend( self.freqs, self.repro_dim, self.repro_idxs['male'] )
            #~ self.freqs = sum_along_axes( females * males * R * SR.extended() * TP.extended(), self.repro_idxs['offspring'] )
            self.freqs = sum_along_axes( females * males * \
                                         REPRO_CONST * \
                                         REPRO_DYN, self.repro_idxs['offspring'] )
            self.normalize()
            
            self.generation += 1
            progress.update(self.generation)
            still_changing = utils.diff(self.freqs, previous) > thresh
        
        self.eq = not still_changing
        if self.runstore != None:   # store final state
            self.runstore.dump_data(self)
            if self.eq:
                state_desc = 'eq'
            else:
                state_desc = 'max'
            self.runstore.record_special_state(self.generation, state_desc)

        # return ProgressBar instance so we can reuse it for further running:
        return progress
