import h5py, datetime
import numpy as np
import utilities as utils
import visualization as viz
for mod in [utils, viz]:
    reload(mod)

def timestamp():
    return "_".join( str( datetime.datetime.now() ).split() )

class Runstore(object):
    def __init__(self, filename, snum=None, rnum=None):
        """
        Open HDF5 file (create if it does not yet exist).
        
        If scenario number `snum` is given, the corresponding group is
        opened. The same goes for run number `rnum`.
    
        Args:
            snum: int
                scenario number
            rnum: int
                simulation run number
        """
        self.f = h5py.File(filename, 'a')
        self.filename = filename
        self.scenario = None
        self.snum = None
        self.run = None
        self.rnum = None
        self.counter = None
        self.gens = None
        self.freqs = None
        self.loci = None
        self.alleles = None
        self.shape = None
        if snum:
            self.scenario = self.select_scenario(snum)
            if rnum:
                self.run = self.select_run(rnum)
    
    def reset(self):
        self.scenario = None
        self.snum = None
        self.run = None
        self.rnum = None
        self.counter = None
        self.gens = None
        self.freqs = None
        self.loci = None
        self.alleles = None
        self.shape = None
    
    def close(self):
        self.f.close()
        self.reset()
    
    def flush(self):
        """
        Flush memory to file.
        """
        self.f.flush()
    
    def open(self, filename, snum=None, rnum=None):
        self.f = h5py.File(filename, 'a')
        self.filename = filename
        self.reset()
        if snum:
            self.scenario = self.select_scenario(snum)
            if rnum:
                self.run = self.select_run(rnum)
        
    def create_scenario(self, snum, labels, desc=None):
        """
        Args:
            snum: int
                scenario number
            labels: tuple
                tuple of loci list and alleles list
            desc: string
                scenario description
        """
        sname = 'scenario_{0}'.format(snum)
        if not sname in self.f:
            self.scenario = self.f.create_group(sname)
            self.loci, self.alleles = labels
            self.scenario['loci'] = np.array(self.loci)    # create dataset (ndarray of strings)
            self.scenario.create_group('alleles')
            for i,loc in enumerate(self.loci):
                self.scenario['alleles'][loc] = self.alleles[i]   # create a dataset for each locus
            self.scenario.attrs['npops'] = len(self.alleles[0])
            self.scenario.attrs['alleleshape'] = utils.list_shape(self.alleles)
            self.scenario.attrs['timestamp'] = timestamp()
            self.snum = snum
            if desc:
                self.write_description(snum, desc)
        else:
            raise KeyError('{0} already exists. You can select it by calling `select_scenario({1})`.'.format(sname,snum))
    
    def init_run(self, rnum, pars, fshape, init_len=100):
        if self.scenario == None:
            raise KeyError('select a scenario first')
        rname = 'run_{0}'.format(rnum)
        if not rname in self.scenario:
            self.run = self.scenario.create_group(rname)
            self.rnum = rnum
            # setting simulation parameters as attributes:
            for p,v in pars.items():
                self.run.attrs[p] = v
            self.run.attrs['timestamp'] = timestamp()
            # integer counter:
            self.counter = self.run.create_dataset('counter', (), 'i')
            # resizable generation array:
            self.gens = self.run.create_dataset('generations', (init_len,), 'i', maxshape=(None,))
            # resizable frequencies array
            self.freqs = self.run.create_dataset('frequencies', (init_len,)+fshape, 'f', maxshape=(None,)+fshape)
            self.sums = self.run.create_group('sums')
            npops = self.scenario.attrs['npops']
            ashape = self.scenario.attrs['alleleshape']
            for i,loc in enumerate(self.loci[1:]):
                ds = self.run['sums'].create_dataset(loc, (init_len,npops,ashape[i+1]), 'f', maxshape=(None,npops,ashape[i+1]))   # create a dataset for each locus
            self.shape = fshape
        else:
            raise KeyError('`{0}` already exists. You can select it by calling `select_run({1})`.'.format(rname,rnum))
    
    def remove_run(self, rnum):
        del self.scenario['run_{0}'.format(rnum)]
        self.run = None
        self.rnum = None
        self.counter = None
        self.gens = None
        self.freqs = None
        self.shape = None
    
    def select_scenario(self, snum, verbose=True):
        if verbose:
            print 'selecting scenario {0} from file {1}'.format(snum, self.filename)
        self.scenario = self.f['scenario_{0}'.format(snum)]
        self.snum = snum
        self.loci = list(self.scenario['loci'])
        self.alleles = list(self.scenario['alleles'])
        return self.scenario
    
    def select_run(self, rnum, verbose=True):
        if verbose:
            print 'selecting run {0} from scenario {1} in file {2}'.format(rnum, self.snum, self.filename)
        self.run = self.scenario['run_{0}'.format(rnum)]
        self.rnum = rnum
        self.counter = self.run['counter']
        self.gens = self.run['generations']
        self.freqs = self.run['frequencies']
        self.shape = self.freqs.shape[1:]
        return self.run
    
    def write_description(self, snum, description):
        """
        Write `description` to user space of scenario `snum`.
        
        Args:
            snum: int
                scenario number
            description: string
                description of the scenario
        """
        if snum != self.snum:
            self.select_scenario(snum, verbose=False)
        desc = self.scenario.create_dataset('description', (), h5py.special_dtype(vlen=str))
        desc[()] = description
    
    def info(self, verbose=False):
        s = 'file: {0}\n'.format(self.filename)
        if self.snum != None:
            s += 'current scenario: {0}\n'.format(self.snum)
            if verbose:
                try:
                    desc = self.scenario['description'][()]
                except KeyError:
                    desc = 'not available'
                s += '[\ndescription: {0}\n]\n'.format(desc)
            if self.rnum != None:
                s += 'current simulation run: {0}\n[\ncount: {1}  (generation: {2})\nfrequency shape: {3}\n]\n'.format(self.rnum, self.counter[()], self.get_generation(), self.shape)
            else:
                s += 'no simulation run selected\n'
        else:
            s += 'no scenario selected\n'
        return s
    
    def full_info(self):
        return self.info(verbose=True)
        
    def advance_counter(self):
        self.counter[()] += 1
    
    def get_count(self):
        return self.counter[()]
    
    def get_generation(self):
        c = self.get_count()
        return self.gens[c-1]
    
    def get_allele_list(self):
        """
        Returns nested list of alleles (excluding `pops`!)
        """
        alleles = []
        for loc in self.loci[1:]:
            alleles.append( list(self.scenario['alleles'][loc][:]) )
        return alleles
    
    def resize(self):
        l = len(self.gens)
        self.gens.resize( (l+100,) )
        self.freqs.resize( (l+100,)+self.shape )
        npops = self.scenario.attrs['npops']
        ashape = self.scenario.attrs['alleleshape']
        for i,loc in enumerate(self.loci[1:]):
            self.run['sums'][loc].resize( (l+100,npops,ashape[i+1]) )
    
    def insert_sums(self, c, locisums):
        for i,loc in enumerate(self.loci[1:]):
            self.run['sums'][loc][c] = locisums[i]
    
    def dump_data(self, metapop):  # gen, freqs, sums):
        """
        Append generation `gen`, frequencies `freqs`, and locus sums
        `sums` to current datasets.
        
        Args:
            gen: int
                generation
            freqs: ndarray
                frequencies
            sums: list of ndarrays
                list of locus sums
        """
        if metapop.generation > 0 and metapop.generation <= self.get_generation():   # dump data must not lie in the past
            pass
        else:
            c = self.get_count()
            if c >= len(self.gens):
                self.resize()
            self.gens[c] = metapop.generation   #gen
            self.freqs[c] = metapop.freqs
            self.insert_sums(c, metapop.all_sums())
            self.advance_counter()
    
    def plot_sums(self):
        pops = list(self.scenario['alleles']['population'][:])
        print pops
        alleles = self.get_allele_list()
        print alleles
        loci = self.loci[1:]
        print loci
        figs = viz.create_figs(pops, loci)
        gens = self.gens[:]
        sums = self.run['sums']
        c = self.get_count()
        viz.plot_sums(gens, sums, c, loci, alleles, figs, lw=2)
        return figs

def get_frequencies(g, filename, snum, rnum):
    with h5py.File(filename, 'r') as df:    # read-only
        run = df['/scenario_{0}/run_{1}'.format(snum,rnum)]
        generations = run['generations'][:]    # generations array
        idx = np.searchsorted(generations, g)   # get index of generation closest to given g
        return run['frequencies'][idx]
