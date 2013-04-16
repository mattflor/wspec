"""
.. module:: utilities
   :platform: Unix
   :synopsis: Utility functions

.. moduleauthor:: Matthias Flor <matthias.c.flor@gmail.com>

"""
import numpy as np
from numpy import sum
import matplotlib.pyplot as plt
from pprint import PrettyPrinter

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

def myfloat(x, threshold=1e-4, absolute_threshold=1e-10):
    if x < absolute_threshold: return '    ---'
    elif x < threshold: return '    0.0'
    else: return '%.4f' % x


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
