"""
.. module:: utilities
   :platform: Unix
   :synopsis: Utility functions

.. moduleauthor:: Matthias Flor <matthias.c.flor@gmail.com>

"""
import numpy as np
from numpy import sum
import matplotlib.pyplot as plt
import pandas as pd
from pprint import PrettyPrinter

def loci2string(loci, alleles):
    loci = ['locus'] + loci
    alleles = ['alleles'] + [', '.join(row) for row in alleles]     # turn list of alleles into a list of strings
    w1 = len(max(loci, key=len))                      # max locus width
    w2 = len(max(alleles, key=len))
    ret = "%-*s      %-*s\n" % (w1, loci[0], w2, alleles[0])    # header row
    ret += '-'*(w1+6+w2)+'\n'
    for loc,allele in zip(loci[1:], alleles[1:]):
        ret += "%-*s      %-*s\n" % (w1, loc, w2, allele)
    return ret.rstrip()

def params2string(params):
    names = ['parameter'] + sorted(params.keys())
    values, descriptions = ['value'], ['description']
    for name in names[1:]:
        v,d = params[name]
        values.append(v)
        descriptions.append(d)
    values = [str(v) for v in values]
    w1 = len(max(names, key=len))
    w2 = len(max(values, key=len))
    w3 = len(max(descriptions, key=len))
    ret = "%-*s    %-*s      %-*s\n" % (w1, names[0], w2, values[0], w3, descriptions[0])
    ret += '-'*(w1+4+w2+6+w3)+'\n'
    for name,value,desc in zip(names[1:], values[1:], descriptions[1:]):
        ret += "%-*s    %-*s      %-*s\n" % (w1, name, w2, value, w3, desc)
    return ret.rstrip()
        
def add_preferences(params, prefs):
    for pref,vdict in sorted(prefs.items()):
        for cue,val in sorted(vdict.items()):
            params['pr_{0}_{1}'.format(pref,cue).lower()] = (val, 'rejection probability')
    return params

def configure_locals(LOCI, ALLELES, parameters):
    config = {}
    config['LOCI'] = LOCI
    config['ALLELES'] = ALLELES
    config['ADICT'] = make_allele_dictionary(LOCI, ALLELES)
    #~ config['LABELS'] = panda_index(ALLELES, LOCI)   # use this as index for conversion of freqs to pd.Series
    config['FSHAPE'] = list_shape(ALLELES)          # shape of frequencies
    repro_axes = reproduction_axes(LOCI)
    config['REPRO_AXES'] = repro_axes  # axes for the reproduction step, used for automatic extension of arrays to the correct shape by inserting np.newaxis
    config['N_LOCI'] = len(LOCI)
    pops = ALLELES[0]
    config['POPULATIONS'] = pops              # shortcut for faster access to populations
    config['N_POPS'] = len(pops)             # number of populations within metapopulations
    config['REPRO_DIM'] = len(repro_axes)
    for name,(value,desc) in parameters.items():
        config[name] = value
    return config

def list_shape(list2d):
    """
    Return the `shape` of a 2-dimensional nested list.
    
    Args:
        list2d: nested list
        
    Returns:
        out: shape tuple
    """
    shape = [0 for i in range(len(list2d))]
    for i,l in enumerate(list2d):
        shape[i] = len(l)
    return tuple(shape)

def extend(arr, dim, pos):
    """
    Broadcast array `arr` to new extended dimension `dim`.
    
    This is achieved by inserting the appropriate number of new axes.
    The original axes of `arr` become positioned at `pos`. Thus, the 
    list `pos` must have length equal to `arr.ndim`.
    
    Args:
        arr: ndarray
        dim: int
        pos: int or list of ints
    Returns:
        out: ndarray
    """
    if isinstance(arr, float):
        return arr
    indexer = [np.newaxis] * dim
    if isinstance(pos,int): pos = [pos]       # enable passing of a single int posistion
    for p in pos:
        indexer[p] = slice(None)
    return arr[indexer]

def sum_along_axes(arr, axes):
    """
    Sum along multiple axes.
    
    Args:
        arr: ndarray
            Input array.
        axes: integer or list of integers
            Axes along which `arr` is summed.
    
    Returns:
        out: ndarray
            Output array. The shape of `out` is identical to the 
            shape of `arr` along `axes`.
    """
    if isinstance(axes,int): axes = [axes]       # enable passing of a single int axis
    _axes = range(arr.ndim)
    for a in axes: _axes.remove(a)
    return np.apply_over_axes(sum, arr, _axes).squeeze()

def sum_over_axes(arr, axes):
    if isinstance(axes,int): axes = [axes]       # enable passing of a single int axis
    return np.apply_over_axes(sum, arr, axes).squeeze()
        
def panda_index(labels, names=None, dtype='|S10'):
    """
    Create a pandas.MultiIndex with row names contained in the nested 
    list `labels` and column names contained in the optional list 
    `names`.
    
    Args:
        labels: nested list of strings
        names: list of strings
    
    Example usage:
        >>> labels = [['wine','water','beer'], [0.2','0.5'], ['to go','for here']]
        >>> names = ['beverage','size','order']
        >>> index = make_index(labels,names)
        >>> index
        
    """
    if names==None:
        names = ['axis{0}'.format(i) for i in range(len(labels))]
    else:
        assert len(labels)==len(names)
    sh = list_shape(labels)
    n_axes = len(labels)
    n_total = np.prod(sh)
    ctile = np.concatenate( ([1],np.cumprod(sh)[:-1]) )
    crep = np.concatenate( (np.cumprod(sh[::-1])[:-1][::-1],[1]) )
    replabels = np.empty((n_axes,n_total), dtype=dtype)
    for i,l in enumerate(labels):
        replabels[i] = np.tile( np.repeat(l,crep[i]), ctile[i] )
    tuples = zip(*replabels)
    return pd.MultiIndex.from_tuples(tuples, names=names)

def myfloat(x, threshold=1e-4, absolute_threshold=1e-10):
    import pandas as pd
    if x < absolute_threshold: return '    ---'
    elif x < threshold: return '    0.0'
    else: return '%.4f' % x
try:
    pd.set_option('display.float_format',myfloat)
except:
    pd.set_printoptions(precision=5)

class MyPrettyPrinter(PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if isinstance(object, float):
            return ('%.4f' % object), True, False
        else:
            return PrettyPrinter.format(self, object, context,
                                        maxlevels, level)

def make_allele_dictionary(loci, alleles):
    adict = {}    # dictionary for allele name to index conversion
    for i,locus in enumerate(alleles):
        for allele in locus:
            adict[allele] = (i,locus.index(allele))
    return adict

def reproduction_axes(loci, who=['female','male','offspring']):
    """
    Create a list of reproduction axes names.
    
    Args:
        loci: list of strings
            names of loci
        who: list of strings
            Can't really think of anything else than the default 
            that would make sense here.
    
    Returns:
        out: list of strings
    """
    if isinstance(who, str):
        who = [who]
    return [loci[0]] + ["{0}_{1}".format(i, locus) for i in who for locus in loci[1:]]

def nuclear_inheritance(n1, n2=None, r=0.5):
    if n2 is not None:
        return nuclear_inheritance_at_two_loci(n1=n1, n2=n2, r=r)
    else:
        return nuclear_inheritance_at_single_locus(n1)

def nuclear_inheritance_at_single_locus(n):
    """Returns an array for the inheritance at a nuclear locus with n alleles."""
    ary = np.zeros((n,n,n))
    for female in range(n):
        for male in range(n):
            for offspring in range(n):
                if female==male==offspring:
                    ary[female,male,offspring]=1.
                if female!=male:
                    if (offspring==female) or (offspring==male):
                        ary[female,male,offspring]=0.5
    return ary

def nuclear_inheritance_at_two_loci(n1,n2,r):
    ary = np.zeros( (n1,n2, n1,n2, n1,n2), float )
    for i in range(n1):
        for j in range(n2):
            for k in range(n1):
                for l in range(n2):
                    for m in range(n1):
                        for n in range(n2):
                            if i==k==m and j==l==n:
                                ary[i,j,k,l,m,n] = 1.
                            if i==k==m and j!=l:
                                if j==n or l==n:
                                    ary[i,j,k,l,m,n] = 0.5
                            if i!=k and j==l==n:
                                if i==m or k==m:
                                    ary[i,j,k,l,m,n] = 0.5
                            if i!=k and j!=l:
                                if (i==m and j==n) or (k==m and l==n):
                                    ary[i,j,k,l,m,n] = 0.5 * (1-r)
                                elif (i==m and l==n) or (k==m and j==n):
                                    ary[i,j,k,l,m,n] = 0.5 * r
    return ary
    
def diff(a, b):
    """
    Sum over absolute differences between two arrays `a` and `b`.
    
    Args:
        a, b: ndarrays
    
    Returns:
        out: float
    """
    return np.sum(np.abs(a-b))

def make_reproduction_allele_names(axes, config):
    """
    Args:
        axes: list of strings
        config: dict
        
    Returns:
        out: nested list of strings
    """
    loci = config['LOCI']
    alleles = config['ALLELES']
    if axes[0] == 'population':
        result = alleles[:1]
        axes =  axes[1:]
    else:
        result = []
    for ax in axes:
        who,locus = ax.split('_')     # `who`: 'female', 'male', or 'offspring'
        w = who[0]                    # take first letter, i.e. 'f', 'm', or 'o'
        als = alleles[loci.index(locus)]
        result.append( ["{0}{1}".format(w,a) for a in als] )
    return result

def get_alleles(loci, config):
    """
    Args:
        loci: list of stings
        config: dict
    
    Returns:
        out: nested list of ints
    """
    return [config['ALLELES'][config['LOCI'].index(locus)] for locus in loci]
