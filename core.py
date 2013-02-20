import numpy as np
import panda as pd
from utilities import *

class ReproductionWeight(object):
    def __init__(self, name, axes, config, unstack_levels=[], **parameters):
        self.name = name
        self.axes = axes
        self.repro_dim = config['REPRO_DIM']
        self.repro_axes = config['REPRO_AXES']
        self.unstack_levels = unstack_levels
        self.parameters = parameters
        self.__dict__.update(parameters)
        self.idxs = [self.repro_axes.index(ax) for ax in axes]
        self.alleles = make_reproduction_allele_names(axes, config)
        self.pd_idx = panda_index( self.alleles, axes )
        self.shape = list_shape( self.alleles )
    
    def set(self, arr):
        assert np.shape(arr) == self.shape
        self.array = arr
        self.panda = pd.Series(self.array.flatten(), index=self.pd_idx, name=self.name)
        
    def update(self, arr=None):
        """
        Set weight array to `arr`.
        
        Args:
            arr: numpy array
        """
        assert self.shape == arr.shape
        self.array = arr
        self.panda.data = self.array
    
    def extended(self):
        """
        Extend weight array dimensions to the correct dimensions for
        the reproduction step.
        """
        return extend(self.array, self.repro_dim, self.idxs)
    
    def set_to_ones(self):
        self.array = np.ones(self.shape,float)
    
    def dynamic(self, arr):
        """
        This method must be overloaded with custom specifics.
        """
        pass
    
    def __str__(self):
        if self.unstack_levels:
            s = '{0}\nName: {1}\n'.format( self.panda.unstack(self.unstack_levels), self.name )
        else:
            s = str(self.panda) + '\n'
        pars = ''
        for k,v in self.parameters.items():
          pars += '{0}: {1}\n'.format(k,v)
        return s + pars.rstrip()
    
    def str_myfloat(self):
        if self.unstack_levels:
            s = '{0}\nName: {1}\n'.format( self.panda.unstack(self.unstack_levels).to_string(float_format=myfloat), self.name )
        else:
            s = '{0}\n'.format( self.panda.to_string(float_format=myfloat) )
        pars = ''
        for k,v in self.parameters.items():
          pars += '{0}: {1}\n'.format(k,v)
        return s + pars.rstrip()

class Metapopulation(object):
    def __init__(self, frequencies, config, generation=0, name='metapopulation', eq='undetermined'):
        self.loci = config['LOCI']
        self.n_loci = len(self.loci)
        self.alleles = config['ALLELES']
        self.repro_axes = config['REPRO_AXES']  # reproduction_axes(loci)
        self.repro_dim = config['REPRO_DIM']    #len(self.repro_axes)
        assert np.shape(frequencies) == list_shape(self.alleles)
        self.freqs = frequencies
        self.ndim = self.freqs.ndim
        self.shape = self.freqs.shape
        self.size = self.freqs.size
        self.normalize()
        self._allele_idxs = make_allele_dictionary(self.loci, self.alleles)
        self.labels = panda_index(self.alleles, self.loci)
        self.populations = self.alleles[0]
        self.n_pops = len(self.populations)
        self.generation = generation
        self.name = name
        self.eq = eq
        self.panda = pd.Series(self.freqs.flatten(), index=self.labels, name=name)
        self.livechart = False
        
        self.male_axes = reproduction_axes(self.loci, 'male')
        self.male_idxs = [self.repro_axes.index(a) for a in self.male_axes]
        self.female_axes = reproduction_axes(self.loci, 'female')
        self.female_idxs = [self.repro_axes.index(a) for a in self.female_axes]
        self.offspring_axes = reproduction_axes(self.loci, 'offspring')
        self.offspring_idxs = [self.repro_axes.index(a) for a in self.offspring_axes]
    
    def __str__(self):
        if not self.isuptodate():
            self.update()
        s = "{0}\nName: {1}\nGeneration: {2}\nEQ: {3}".format( \
                self.panda.unstack([0,-1]).to_string(float_format=myfloat), \
                self.name, \
                self.generation, \
                self.eq )
        return s
    
    def overview(self):
        s = str(self.get_sums_pd([1,2]).unstack(2)) + '\n'
        s += 'Name: background loci\n\n'
        for loc in self.loci[3:]:
            s += str(self.get_sums_pd(loc).unstack(1)) + '\n'
            s += 'Name: {0}\n\n'.format(loc)
        return s
    
    def normalize(self):
        s = sum_along_axes(self.freqs, 0)          # first axis are `populations`
        self.freqs /= extend(s, self.ndim, 0)      # in-place, no copy
    
    def isuptodate(self):
        return np.all(self.panda.values == self.freqs.flatten())
    
    def update(self):
        self.panda.data = self.freqs.flatten()
    
    def store_freqs(self, filename='freqs.npy'):
        np.save(filename, self.freqs)
    
    def load_freqs(self, filename='freqs.npy'):
        freqs = np.load(filename)
        assert np.shape(freqs) == self.shape
        self.freqs = freqs
        self.update()
    
    def load(self, frequencies, generation):
        self.generation = generation
        self.freqs = frequencies[str(generation)]
    
    def get_sum(self, allele, pop):
        if not isinstance(pop,int): pop = self.populations.index(pop)
        l,a = self._allele_idxs[allele]
        return sum_along_axes(self.freqs, [0,l])[pop,a]

    def get_sums(self, locus, pop=None):
        level = [0]
        if not isinstance(locus, list):
            locus = [locus]
        for loc in locus:
            if isinstance(loc, int): level.append(loc)
            else: level.append( self.loci.index(loc) )
        if pop or pop==0:
            if not isinstance(pop,int):
                popname, pop = pop, self.populations.index(pop)
            else:
                popname = self.populations[pop]
            return sum_along_axes(self.freqs, level)[pop]
        return sum_along_axes(self.freqs, level)
    
    def get_sums_pd(self, locus, pop=None):
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
        `pop` : population index or name
        `allele` : allele name
        `intro_freq` : introduction frequency of `allele`
        """
        if not isinstance(pop,int):
            pop = self.populations.index(pop)
        loc,al = self._allele_idxs[allele]
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
    
    def run(self, weights, n=1000, step=100, threshold=1e-4, chart=None):
        """
        Simulate next `n` generations. Abort if average overall difference 
        between consecutive generations is smaller than `threshold`.
        `step` is used for plotting only.
        
        Args:
            weights: iterable of weights to be used in the calculation
                     of the next generation frequencies
        
        To enable live stripcharting, pass a stripchart instance to `chart`.
        """
        M_, V_, R_, SR, TP = weights
        global SR, TP
        self.chart = chart
        n += self.generation
        thresh = threshold/self.size   # on average, each of the frequencies should change less than `thresh`
        pt = SR.pt
        species_preferences = [(sno,prefs) for pname,sno,prefs in SR.preferences]
        trait_preferences = [(pno,prefs) for pname,pno,prefs in TP.preferences]
        still_changing = True
        while still_changing and self.generation < n:
            previous = np.copy(self.freqs)
            ### migration ##################################
            self.freqs = np.sum(self.freqs[np.newaxis,...] * M_, 1)   # sum over source axis
            self.normalize()
            
            ### viability selection ########################
            self.freqs = self.freqs * V_
            self.normalize()
            
            ### reproduction ###############################
            # species recognition:
            SR.set_to_ones()
            ABfreqs = self.get_sums(['backA','backB'])
            for sno, prefs in species_preferences:
                for pop, ano, bno, pr in prefs:
                    R = 1./(1-pr*pt*(1-ABfreqs[pop,ano,bno]))
                    SR.array[pop,sno] *= (1-pr)*R
                    SR.array[pop,sno,ano,bno] = R
            SR.array = np.nan_to_num(SR.array)
            SR_ = SR.extended()
            
            TP.update(self.get_sums('trait'))
            
            # trait preferences:
            TP.set_to_ones()
            traitfreqs = self.get_sums('trait')
            for pno, prefs in trait_preferences:
                for pop, tno, pr in prefs:
                    R = 1./(1-pr*pt*(1-traitfreqs[pop,tno]))
                    TP.array[pop,pno] *= (1-pr)*R
                    TP.array[pop,pno,tno] = R
            TP.array = np.nan_to_num(TP.array)           # replace NaN with zero (happens when pr=pt=1 and x=0)
            TP_ = TP.extended()
            
            TP.update(self.get_sums('trait'))
            TP_ = TP.extended()
            
            # offspring production:
            females = extend( self.freqs, REPRO_DIM, self.female_idxs )
            males = extend( self.freqs, REPRO_DIM, self.male_idxs )
            offspring = sum_along_axes( females * males * R_ * SR_ * TP_, self.offspring_idxs )
            self.freqs = offspring
            self.normalize()
            
            if self.generation % step == 0:
                GENS.append(self.generation)
                FREQS.append(self.freqs)
                allele_freqs = []
                for i,pop in enumerate(self.populations):
                    allele_freqs.append([])
                    for al in chartlabels[pop]:
                        allele_freqs[i].append( self.get_sum(al, pop) )
                update_plot_data(self.generation, allele_freqs)
                if self.chart:
                    self.chart.update()
            
            self.generation += 1
            still_changing = diff(self.freqs, previous) > thresh
  
        self.eq = not still_changing
        if self.chart:
            self.chart.finalize()
