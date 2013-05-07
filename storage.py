import datetime, h5py
import numpy as np
import utilities as utils
import visualization as viz
for mod in [utils, viz]:
    reload(mod)

def timestamp():
    return "_".join( str( datetime.datetime.now() ).split() )

def empty_current():
    d = {'scenario': None,    # scenario related shortcuts
         'sid': None,        #   |
         'loci': None,        #   V
         'alleles': None,     # 
         'description': None, # _____________________
         'run': None,         # run related shortcuts
         'rid': None,        #   |
         'count': None,       #   V
         'generation': None,  #
         'gens': None,        #
         'freqs': None,       #
         'fshape': None,      #
         'sums': None         # ______________________
         }
    return d

special_dtype = np.dtype([('generation', 'i'), ('description', 'S64')])
par_dtype = np.dtype([('name', 'S16'), ('value', 'f')])
long_par_dtype = np.dtype([('name', 'S16'), ('value', 'f'), ('description', 'S128')])

class RunStore(object):
    def __init__(self, filename, sid=None, rid=None):
        """
        Open HDF5 file (create if it does not yet exist).
        
        If scenario id `sid` is given, the corresponding group is
        opened. The same goes for run id `rid`.
    
        Args:
            sid: int
                scenario id
            rid: int
                simulation run id
        """
        self.f = h5py.File(filename, 'a')
        self.filename = filename
        self.counter = None
        self.current = empty_current()
        self.select_scenario(sid, verbose=False)
        self.select_run(rid, verbose=False)
    
    def update_current(self, **kwargs):
        self.current.update(kwargs)
    
    def reset(self):
        self.counter = None
        self.current = empty_current()
    
    def reset_run(self):
        """
        Only reset `run` related variables.
        """
        self.counter = None
        self.update_current(rid=None, run=None, count=None, generation=None, gens=None, freqs=None, fshape=None, sums=None)
    
    def close(self):
        self.f.close()
        self.reset()
    
    def flush(self):
        """
        Flush memory to file.
        """
        self.f.flush()
    
    def open(self, filename=None, sid=None, rid=None):
        """
        If no filename is provided, the last one used will e re-used.
        """
        if filename is not None:
            self.filename = filename
        self.f = h5py.File(self.filename, 'a')
        self.reset()
        self.select_scenario(sid, verbose=False)
        self.select_run(rid, verbose=False)
        
    def create_scenario(self, sid, labels, description=None):
        """
        Args:
            sid: int
                scenario id
            labels: tuple
                tuple of loci list and (nested) alleles list
            description: string
                scenario description
        """
        sname = 'scenario_{0}'.format(sid)
        if not sname in self.f:
            scenario = self.f.create_group(sname)
            loci, alleles = labels
            scenario['loci'] = np.array(loci)    # create dataset (ndarray of strings)
            scenario.create_group('alleles')
            for i,loc in enumerate(loci):
                scenario['alleles'][loc] = alleles[i]   # create a dataset for each locus
            scenario.attrs['npops'] = len(alleles[0])
            scenario.attrs['alleleshape'] = utils.list_shape(alleles)
            scenario.attrs['timestamp'] = timestamp()
            if description is None:
                description = 'no description available'
            desc = scenario.create_dataset('description', (), h5py.special_dtype(vlen=str))
            desc[()] = description
            self.update_current(sid=sid, scenario=scenario, loci=loci, alleles=alleles, description=description)
            self.reset_run()         # if a `run` was previously selected, it must be reset now to avoid inconsistencies
            return scenario
        else:
            raise KeyError('{0} already exists. You can select it by calling `select_scenario({1})`.'.format(sname,sid))
        self.flush()
    
    def init_run(self, rid, pars, fshape, init_len=100):
        scenario = self.current['scenario']
        if scenario is None:
            raise KeyError('select a scenario first')
        rname = 'run_{0}'.format(rid)
        if rname not in scenario:
            run = scenario.create_group(rname)
            # timestamp as attribute:
            run.attrs['timestamp'] = timestamp()
            # parameters:
            npars = len(pars)   # number of paramters
            #~ parameters = run.create_dataset('parameters', (npars,), par_dtype)
            parameters = run.create_dataset('parameters', (npars,), long_par_dtype)
            i = 0
            # backwards compatibility:
            ks = pars.keys()
            if not isinstance(pars[ks[0]], tuple):
                for name,value in sorted(pars.items()):
                    parameters[i] = (name, value, '')
                    i += 1
            # end of backward comp.
            else:
                for name,(value,desc) in sorted(pars.items()):
                    parameters[i] = (name, value, desc)
                    i += 1
            # integer counter:
            self.counter = run.create_dataset('counter', (), 'i')
            # resizable generation array:
            gens = run.create_dataset('generations', (init_len,), 'i', maxshape=(None,))
            special_states = run.create_dataset('special states', (1,), special_dtype, maxshape=(None,))
            # resizable frequencies array
            freqs = run.create_dataset('frequencies', (init_len,)+fshape, 'f', maxshape=(None,)+fshape)
            npops = scenario.attrs['npops']
            ashape = scenario.attrs['alleleshape']
            sums = run.create_group('sums')
            for i,loc in enumerate(scenario['loci'][1:]):
                ds = run['sums'].create_dataset(loc, (init_len,npops,ashape[i+1]), 'f', maxshape=(None,npops,ashape[i+1]))   # create a dataset for each locus
            self.update_current(rid=rid, run=run, gens=gens, freqs=freqs, fshape=fshape, sums=sums)
            return run
        else:
            raise KeyError('`{0}` already exists. You can select it by calling `select_run({1})`.'.format(rname,rid))
    
    def remove_run(self, rid, sid):
        """
        For safety reasons, one has to specify run AND scenario id.
        """
        scenario = self.get_scenario(sid)
        del scenario['run_{0}'.format(rid)]
        if rid == self.current['rid']:
            self.reset_run()
    
    def get_scenario(self, sid):
        return self.f['scenario_{0}'.format(sid)]
    
    def get_run(self, rid, sid=None):
        if sid is None:
            return self.current['scenario']['run_{0}'.format(rid)]
        else:
            scenario = self.get_scenario(sid)
            return scenario['run_{0}'.format(rid)]
    
    def get_parameters(self):
        """
        Returns a dictionay containing parameters names and values.
        """
        pars = {}
        pdata = self.current['run']['parameters']
        for pname,pval,pdesc in pdata:
            pars[pname] = (pval,pdesc)
        return pars
    
    def select_scenario(self, sid, verbose=True):
        if sid is not None:
            scenario = self.get_scenario(sid)
            loci = list(scenario['loci'][:])
            alleles = self.get_allele_list(sid=sid, with_pops=True)
            try:
                description = scenario['description'][()]
            except KeyError:   # existing scenarios might not have descriptions
                description = 'no description available'
                desc = scenario.create_dataset('description', (), h5py.special_dtype(vlen=str))
                desc[()] = description
            self.update_current(scenario=scenario, sid=sid, loci=loci, alleles=alleles, description=description)
            if verbose:
                print 'selecting scenario {0} from file {1}'.format(sid, self.filename)
            return scenario
        else:
            if verbose:
                print 'please specify a scenario id'
    
    def select_run(self, rid, sid=None, verbose=True):
        if rid is not None:
            if sid is not None:
                self.select_scenario(sid, verbose=verbose)   # this updates `scenario` related current entries and resets all `run` related entries
            run = self.get_run(rid, sid=sid)
            gens = run['generations']
            freqs = run['frequencies']
            fshape = freqs.shape[1:]
            sums = run['sums']
            self.update_current(rid=rid, run=run,  gens=gens, freqs=freqs, fshape=fshape, sums=sums)
            self.counter = run['counter']
            count = self.get_count()
            generation = self.get_last_generation()
            self.update_current(count=count, generation=generation)
            if verbose:
                print 'selecting existing run {0} from scenario {1} in file {2}'.format(rid, self.current['sid'], self.filename)
            return run
        else:
            if verbose:
                print 'please specify a run id'
    
    def update_description(self, description, sid=None):
        if sid is None:
            scenario = self.current['scenario']
        else:
            scenario = self.get_scenario(sid)
        desc = scenario['description']
        desc[()] = description
        if sid == self.current['sid']:
            self.update_current(description=description)
    
    def info(self, sid=None, verbose=False):
        s = 'file: {0}\n'.format(self.filename)
        if sid is None:
            sid = self.current['sid']
        if sid is not None:
            scenario = self.get_scenario(sid)
            loci = list(scenario['loci'][:])
            alleles = self.get_allele_list(with_pops=True)
            s += "selected scenario: {0}\n\tpops:\n\t\t{1}\n\tloci:".format(sid, ', '.join(alleles[0]))
            w = max(loci, key=len)
            loc_format = "\t\t{{0:<{0}s}}\t{{1}}\n".format(w)  # adjust width to longest locus name 
            for i,loc in enumerate(loci[1:]):
                s += loc_format.format(loc+':', ', '.join(alleles[i]))
            if verbose:
                description = scenario['description'][()]
                s += "\tdescription: {0}\n".format(description)
            if self.rid is not None:
                s += 'selected simulation run: {0}\n\tcount / generation: {1} / {2})\n\tfrequency shape: {3}\n]\n'.format(self.rid, self.counter[()], self.get_last_generation(), self.current['fshape'])
            else:
                s += 'no simulation run selected\n'
        else:
            s += 'no scenario selected\n'
        return s
    
    def full_info(self, sid=None):
        return self.info(sid=sid, verbose=True)
        
    def advance_counter(self):
        self.counter[()] += 1
    
    def get_count(self, g=None):
        if g is None:
            return self.counter[()]
        else:
            return self.get_closest_count(g)
            
    def get_closest_count(self, g):
        c = self.get_count()
        return np.argmin(np.abs(self.current['gens'][:c] - g))
    
    def get_last_generation(self):
        c = self.get_count()
        return self.current['gens'][c-1]
        
    def get_closest_generation(self, g):
        """
        Out of all stored generations, returns the one closest to `g`.
        """
        closest = self.get_closest_count(g)
        return self.current['gens'][closest]
    
    def get_frequencies(self, g):
        """
        Returns the metapopulation frequencies of the currently selected
        run for generation g.
        """
        closest = self.get_closest_generation(g)
        c = self.get_count(closest)
        return self.current['run']['frequencies'][c], closest
            
    
    def get_allele_list(self, sid=None, with_pops=False):
        """
        Returns nested list of alleles (w or w/o `pops`)
        """
        if sid is None:    # use current
            scenario = self.current['scenario']
        else:
            scenario = self.get_scenario(sid)
        if with_pops:
            loci = scenario['loci']
        else:
            loci = scenario['loci'][1:]
        alleles = []
        for loc in loci:
            alleles.append( list(scenario['alleles'][loc][:]) )
        return alleles
    
    def resize(self):
        run = self.current['run']
        gens = self.current['gens']
        freqs = self.current['freqs']
        fshape = self.current['fshape']
        l = len(gens)
        gens.resize( (l+100,) )
        freqs.resize( (l+100,)+fshape )
        scenario = self.current['scenario']
        npops = scenario.attrs['npops']
        ashape = scenario.attrs['alleleshape']
        for i,loc in enumerate(self.current['loci'][1:]):
            run['sums'][loc].resize( (l+100,npops,ashape[i+1]) )
        #~ self.update_current(scenario=scenario, run=run, gens=gens, freqs=freqs)
    
    def append_sums(self, c, loci_sums):
        for i,loc in enumerate(self.current['loci'][1:]):
            self.current['run']['sums'][loc][c] = loci_sums[i]
    
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
        if metapop.generation > 0 and metapop.generation <= self.get_last_generation():
            # dump data must not lie in the past or
            # have been stored already
            pass
        else:
            c = self.get_count()
            if c >= len(self.current['gens']):
                self.resize()
            self.current['gens'][c] = metapop.generation   #gen
            self.current['freqs'][c] = metapop.freqs
            self.append_sums(c, metapop.all_sums())
            self.advance_counter() 
    
    def record_special_state(self, g, description):
        """Args:
            g: int
                generation
            description: str (max length: 64)
                usually one of the following state descriptions
                `start`    - start state
                `eq`       - an equilibrium has been reached
                `intro`    - introduction of an allele
                `max`      - the maximum number of generations has been reached
        """
        special_states = self.current['run']['special states']
        # append new state:
        special_states[-1] = (g, description)
        # resize to make room for next state:
        l = len(special_states)
        special_states.resize( (l+1,) )
        self.flush()
    
    def get_special_states(self):
        return self.current['run']['special states'][:-1]
    
    def get_current(self):
        return self.current['scenario'], self.current['run']
    
    def plot_sums(self, figsize=[19,7], **kwargs):
        scenario, run = self.get_current()
        alleles = self.get_allele_list(with_pops=True)
        alleles = alleles[:]
        loci = list(scenario['loci'][:])
        gens = run['generations'][:]
        sums = run['sums']
        c = run['counter'][()]
        fig = viz.plot_sums(gens, sums, c, loci, alleles, figsize=figsize, lw=3, **kwargs)
        return fig
    
    def plot_overview(self, generation, show_generation=True, figsize=[18,5]):
        scenario, run = self.get_current()
        g = self.get_closest_generation(generation)
        sums = self.get_sums(generation)
        loci = list(scenario['loci'][:])
        alleles = self.get_allele_list(with_pops=True)
        if show_generation:
            fig = viz.stacked_bars(sums, loci, alleles, generation=generation, figsize=figsize)
        else:
            fig = viz.stacked_bars(sums, loci, alleles, figsize=figsize)
        return fig

    def get_sums(self, generation):
        scenario, run = self.get_current()
        c = self.get_closest_count(generation)
        sums = run['sums']
        result = []
        for loc in scenario['loci'][1:]:
            result.append(sums[loc][c])
        return result
        

def get_frequencies(g, filename, sid, rid):
    with h5py.File(filename, 'r') as df:    # read-only
        run = df['/scenario_{0}/run_{1}'.format(sid,rid)]
        generations = run['gens'][:]    # generations array
        idx = np.searchsorted(generations, g)   # get index of generation closest to given g
        return run['freqs'][idx]
