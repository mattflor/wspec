import numpy as np
import panda as pd
from utilities import *

class reproduction_weight(object):
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
        if arr:
            self.array = arr
        self.panda.data = self.array
    
    def extended(self):
        return extend(self.array, self.repro_dim, self.idxs)
    
    def set_to_ones(self):
        self.array = np.ones(self.shape,float)
    
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
