import h5py


class runstore(object):
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
        self.shape = None
    
    def close(self):
        self.f.close()
        self.reset()
    
    def open(self, filename, snum=None, rnum=None):
        self.f = h5py.File(filename, 'a')
        self.filename = filename
        self.reset()
        if snum:
            self.scenario = self.select_scenario(snum)
            if rnum:
                self.run = self.select_run(rnum)
        
    def create_scenario(self, snum, desc=None):
        sname = 'scenario_{0}'.format(snum)
        if not sname in self.f:
            self.scenario = self.f.create_group(sname)
            self.snum = snum
            if desc:
                self.write_description(snum, desc)
        else:
            raise KeyError('{0} already exists. You can select it by calling `select_scenario({1})`.'.format(sname,snum))
    
    def init_run(self, rnum, pars, fshape):
        if self.scenario == None:
            raise KeyError('select a scenario first')
        rname = 'run_{0}'.format(rnum)
        if not rname in self.scenario:
            self.run = self.scenario.create_group(rname)
            self.rnum = rnum
            # setting simulation parameters as attributes:
            for p,v in pars.items():
                self.run.attrs[p] = v
            # integer counter:
            self.counter = self.run.create_dataset('counter', (), 'i')
            # resizable generation array:
            self.gens = self.run.create_dataset('generations', (100,), 'i', maxshape=(None,))
            # resizable frequencies array
            self.freqs = self.run.create_dataset('frequencies', (100,)+fshape, 'i', maxshape=(None,)+fshape)
            self.shape = fshape
        else:
            raise KeyError('`{0}` already exists. You can select it by calling `select_run({1})`.'.format(rname,rnum))
            
    
    def select_scenario(self, snum, verbose=True):
        if verbose:
            print 'selecting scenario {0} from file {1}'.format(snum, self.filename)
        self.scenario = self.f['scenario_{0}'.format(snum)]
        self.snum = snum
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
                s += 'current simulation run: {0}\n[\ncount: {1}\nfrequency shape: {2}\n]\n'.format(self.rnum,self.counter[()],self.shape)
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
    
    def resize(self):
        l = len(self.gens)
        self.gens.resize((l+100,))
        self.freqs.resize((l+100,)+self.shape)
    
    def dump_data(self, gen, freqs):
        """
        Append generation `gen` and frequencies `freqs` to current datasets.
        """
        c = self.get_count()
        if c >= len(self.gens):
            self.resize()
        self.gens[c] = gen
        self.freqs[c] = freqs
        self.advance_counter()
