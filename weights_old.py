import copy, cPickle, pdb
import numpy as np
from pywolb.utilities import numpytools
from pywolb import base


for module in [numpytools,base]:
    reload(module)

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

def nuclear_inheritance_at_y_locus(n):
    """
    Returns an array for the inheritance at a y-chromosome locus with n alleles 
    (where the first allele means "no y-chromosome", i.e. female).
    """
    ary = np.zeros((n,n,n))      # female, male, offspring
    for male in range(1,n):
        ary[0,male,0] = ary[0,male,male] = 1        # one daughter, one son
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

def offspring_fertility(n1=4,n2=4):
    """Compatible index groups 0/1 (A1,A2,B1,B2) and 2/3 (A3,A4,B3,B4)
    
    To determine the appropriate marker state, we will count the number
    of alleles (during the diploid phase) belonging to index group 1.
    
    Gr. 1   Steril.
    count   marker  Explanation
    ---------------------------------------------------------------------
    0 or 4  S0      no allele involved in incompatibilities, fully fertile (1)
    2       S1      all 4 alleles involved in incompatibilities, sterile (1-h)   
    1 or 3  S2      3 alleles involved in incompatibilities, partially sterile (1-d*h)"""
    ary = np.zeros( (n1,n2, n1,n2, 3), float )
    for i in range(n1):
        for j in range(n2):
            for k in range(n1):
                for l in range(n2):
                    # set group counter to zero (one counter is sufficient)
                    gc1 = 0
                    for index in [i,j,k,l]:            
                        if index in [0,1]: gc1+=1
                    if gc1==0 or gc1==4:
                        ary[i,j,k,l,0] = 1.   # set mark at S0
                    elif gc1==1 or gc1==3:
                        ary[i,j,k,l,2] = 1.   # set mark at S2
                    else:
                        ary[i,j,k,l,1] = 1.   # set mark at S1
    return ary

def element_labels(hosts, *loci):
    """Returns a nested element label list.
    
    Usage: hosts = ['f','m','o']
           hosts = ['f','m']
           loci  = 'T',range(1,3)
           loci  = 'T',range(1,3), 'P',range(2)
    """
    num_loci = int(len(loci) / 2)        # number of loci
    elements = []
    for i in hosts:
        for j in range(num_loci):
            locus_abbrev, allele_range = loci[j*2:(j+1)*2]
            elements.append( ["%s%s%d" % (i,locus_abbrev,k) for k in allele_range] )
    return elements

def element_labels_inheritance(*loci):
    return element_labels(['f','m','o'], *loci)

def axis_labels(hosts, *locus_names):
    """Usage: hosts = ['female', 'male', 'offspring']
              locus_names = 'trait','preference'
    """
    axes = ["%s %s" % (i, name) for i in hosts for name in locus_names]
    return axes

def axis_labels_inheritance(*locus_names):
    return axis_labels(['female','male','offspring'], *locus_names)

class CytoplasmicIncompatibility(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['ci_level']
    labels = dict( axes=['male cytotype', 'offspring cytotype'], \
                   elements=[['mU','mW'],['oU','oW']] )
    _info = 'cytoplasmic incompatibility'
    
    def __init__(self, ci_level, scope='global'):
        base.Weight.__init__(self, ci_level=ci_level, _scope=scope)
    
    def _build(self):
        ary = np.array( [[1, 1], [1-self.ci_level, 1]], float )
        return ary

class ModifiedCI(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['ci_level', 'mod_penetrance']
    labels = dict( axes=['male resistance', 'male cytotype', 'offspring cytotype'], \
                   elements=[['mR0', 'mR1'], ['mU', 'mW'], ['oU', 'oW']] )
    _info = 'cytoplasmic incompatibility modified by male resistance'
    
    def __init__(self, ci_level, mod_penetrance, scope='global'):
        base.Weight.__init__(self, ci_level=ci_level, mod_penetrance=mod_penetrance, _scope=scope)
    
    def _build(self):
        ci = self.ci_level
        e = self.mod_penetrance
        ary = np.array( [[[1,          1], \
                         [1-ci,       1]], \
                        [[1,          1], \
                         [1-(1-e)*ci, 1]]], float )
        return ary

class ModifiedCI2(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['ci_level', 'mod_penetrance']
    labels = dict( axes=['male resistance-Q', 'male resistance-R', 'male cytotype', 'offspring cytotype'], \
                   elements=[['mQ1', 'mQ2'], ['mR1', 'mR2'], ['mU', 'mW'], ['oU', 'oW']] )
    _info = 'cytoplasmic incompatibility modified by 2 male resistance loci Q and R (only Q1R1 provides resistance)'
    
    def __init__(self, ci_level, mod_penetrance, scope='global'):
        base.Weight.__init__(self, ci_level=ci_level, mod_penetrance=mod_penetrance, _scope=scope)
    
    def _build(self):
        ci = self.ci_level
        e = self.mod_penetrance
        ary = np.array( [[[[1,          1], \
                          [1-(1-e)*ci, 1]], \
                         [[1,          1], \
                          [1-ci,       1]]], \
                        [[[1,          1], \
                          [1-ci,       1]], \
                         [[1,          1], \
                          [1-ci,       1]]]], float )
        return ary

class ModifiedCI2CustomLoci(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['ci_level', 'mod_penetrance', 'locus1', 'locus2']
    _info = 'cytoplasmic incompatibility modified by 2 custom male resistance loci locus1 and locus2 (only the right allele combination provides resistance)'
    
    def __init__(self, ci_level, mod_penetrance, scope='global'):
        base.Weight.__init__(self, ci_level=ci_level, mod_penetrance=mod_penetrance, _scope=scope)
    
    def __init__(self, ci_level, mod_penetrance, locus1='Q', locus2='R', scope='global'):
        labels = dict( axes=['male locus-%s' % locus1, 'male locus-%s' % locus2, 'male cytotype', 'offspring cytotype'], \
                   elements=[['m%s1' % locus1, 'm%s2'% locus1], ['m%s1' % locus2, 'm%s2'% locus2], ['mU', 'mW'], ['oU', 'oW']] )
        self.labels = labels
        base.Weight.__init__(self, ci_level=ci_level, mod_penetrance=mod_penetrance, locus1=locus1, locus2=locus2, _scope=scope)        
    
    def _build(self):
        ci = self.ci_level
        e = self.mod_penetrance
        ary = np.array( [[[[1,          1], \
                          [1-(1-e)*ci, 1]],              # mR1-mQ1-mW oU    modified CI case
                         [[1,          1], \
                          [1-ci,       1]]], \
                        [[[1,          1], \
                          [1-ci,       1]], \
                         [[1,          1], \
                          [1-ci,       1]]]], float )
        return ary    

class YModifiedCI(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['ci_level', 'mod_penetrance']
    labels = dict( axes=['male y-chromosome', 'male cytotype', 'offspring cytotype'], \
                   elements=[['mY0', 'mY1', 'mY2'], ['mU', 'mW'], ['oU', 'oW']] )
    _info = 'cytoplasmic incompatibility modified by male resistance on the y-chromosome: Y1 males are resistant'
    
    def __init__(self, ci_level, mod_penetrance, scope='global'):
        base.Weight.__init__(self, ci_level=ci_level, mod_penetrance=mod_penetrance, _scope=scope)
    
    def _build(self):
        ci = self.ci_level
        e = self.mod_penetrance
        ary = np.array( [[[0,          0], \
                         [0,          0]], \
                        [[1,          1], \
                         [1-(1-e)*ci, 1]], \
                        [[1,          1], \
                         [1-ci,       1]]], float )
        return ary

class ConditionalResistanceCosts(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['viability_reduction']
    labels = { 'axes': ['male resistance', 'male cytotype'], \
               'elements': [['mR0', 'mR1'], ['mU', 'mW']] }
    _info = 'Resistant males have reduced viability if they are infected with Wolbachia'
    
    def __init__(self, viability_reduction, scope='global'):
        base.Weight.__init__(self, viability_reduction=viability_reduction, _scope=scope)
    
    def _build(self):
        c = self.viability_reduction
        ary = np.array( [[1, 1], \
                        [1, 1-c]], float )
        return ary
    
class ResistanceCostsY(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['viability_reduction']
    labels = { 'axes': ['male y-chromosome'], \
               'elements': [['mY0', 'mY1', 'mY2']] }
    _info = 'Males carrying the resistant Y chromosome have reduced viability independent of their infection status'
    
    def __init__(self, viability_reduction, scope='global'):
        base.Weight.__init__(self, viability_reduction=viability_reduction, _scope=scope)
    
    def _build(self):
        c = self.viability_reduction
        ary = np.array( [1, 1-c, 1], float )
        return ary

class ResistanceCosts(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['viability_reduction']
    labels = { 'axes': ['male resistance'], \
               'elements': [['mR0', 'mR1']] }
    _info = 'Resistant males have reduced viability independent of their infection status'
    
    def __init__(self, viability_reduction, scope='global'):
        base.Weight.__init__(self, viability_reduction=viability_reduction, _scope=scope)
    
    def _build(self):
        c = self.viability_reduction
        ary = np.array( [1, 1-c], float )
        return ary
    
#class EpistaticResistanceCosts(base.Weight):
#    _type = 'constant viability selection'
#    _parameter_names = ['viability_reduction', 'dominance_level']
#    labels = { 'axes': ['male resistance', 'male background-A', 'male background-B'], \
#               'elements': [['mR0', 'mR1'], \
#                            ['mA1', 'mA2'], \
#                            ['mB1', 'mB2']] }
#    _info = 'Resistence gene causes epistatic problems in the wrong background'
#    
#    def __init__(self, viability_reduction, dominance_level, scope='global'):
#        base.Weight.__init__(self, viability_reduction=viability_reduction, dominance_level=dominance_level, _scope=scope)
#    
#    def _build(self):
#        c = self.viability_reduction
#        k = self.dominance_level
#        ary = np.array( [[[1,     1     ], \
#                         [1,     1     ]], \
#                        [[1,     1-k*c ], \
#                         [1-k*c, 1-c   ]]], float )
#        return ary

class EpistaticResistanceCosts(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['viability_reduction', 'dominance_level']
    labels = { 'axes': ['offspring background-A', 'offspring background-B', 'offspring resistance'], \
               'elements': [['oA1', 'oA2'], \
                            ['oB1', 'oB2'], \
                            ['oR0', 'oR1']] }
    _info = 'Resistence gene causes epistatic problems in the wrong background'
    
    def __init__(self, viability_reduction, dominance_level, scope='global'):
        base.Weight.__init__(self, viability_reduction=viability_reduction, dominance_level=dominance_level, _scope=scope)
    
    def _build(self):
        c = self.viability_reduction
        k = self.dominance_level
        ary = np.array( [[[1., 1.    ], \
                         [1., 1-k*c ]], \
                        [[1., 1-k*c ], \
                         [1., 1-c   ]]], float )
        return ary

class EpistaticResistanceCosts(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['viability_reduction', 'dominance_level']
    labels = { 'axes': ['offspring background-A', 'offspring background-B', 'offspring resistance'], \
               'elements': [['oA1', 'oA2'], \
                            ['oB1', 'oB2'], \
                            ['oR0', 'oR1']] }
    _info = 'Resistence gene causes epistatic problems in the wrong background'
    
    def __init__(self, viability_reduction, dominance_level, scope='global'):
        base.Weight.__init__(self, viability_reduction=viability_reduction, dominance_level=dominance_level, _scope=scope)
    
    def _build(self):
        c = self.viability_reduction
        k = self.dominance_level
        ary = np.array( [[[1., 1.    ], \
                         [1., 1-k*c ]], \
                        [[1., 1-k*c ], \
                         [1., 1-c   ]]], float )
        return ary    

class FecundityReduction(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['fecundity_reduction']
    labels = { 'axes': ['female cytotype'], \
               'elements': [['fU', 'fW']] }
    _info = 'fecundity of females'
    
    def __init__(self, fecundity_reduction, scope='global'):
        base.Weight.__init__(self, fecundity_reduction=fecundity_reduction, _scope=scope)
    
    def _build(self):
        f = self.fecundity_reduction
        ary = np.array( [1, 1-f], float )
        return ary
        
class MaleFertility(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['sterility_coefficient', 'partial_factor']
    labels = { 'axes': ['male sterility-marker'], \
               'elements': [['mS0', 'mS1', 'mS2']] }
    _info = 'fertility/sterility of males'
    
    def __init__(self, sterility_coefficient, partial_factor, scope='global'):
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, partial_factor=partial_factor, _scope=scope)
    
    def _build(self):
        h = self.sterility_coefficient
        d = self.partial_factor
        ary = np.array( [1, 1-h, 1-d*h], float )
        return ary

class SetOffspringFertility(base.Weight):
    _type = 'constant reproduction'
    labels = { 'axes': ['female background-A', 'female background-B', \
                        'male background-A', 'male background-B', \
                        'offspring sterility-marker'], \
               'elements': [["fA%d" % i for i in range(1,5)], \
                            ["fB%d" % i for i in range(1,5)], \
                            ["mA%d" % i for i in range(1,5)], \
                            ["mB%d" % i for i in range(1,5)], \
                            ["oS%d" % i for i in range(3)]] }
    
    def __init__(self, scope='global'):
        base.Weight.__init__(self, _scope=scope)
        
    def _build(self):
        return offspring_fertility(n1=4, n2=4)

class WolbachiaTransmission(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['transmission_rate']
    labels = { 'axes': ['female cytotype', 'offspring cytotype'], \
               'elements': [['fU', 'fW'], ['oU', 'oW']] }
    _info = 'Wolbachia transmission'
    
    def __init__(self, transmission_rate, scope='global'):
        base.Weight.__init__(self, transmission_rate=transmission_rate, _scope=scope)
    
    def _build(self):
        t = self.transmission_rate
        ary = np.array( [[1, 0], [1-t, t]], float )
        return ary

class NuclearInheritanceAtSingleLocus(base.Weight):
    _type = 'constant reproduction'
    
    def __init__(self, name, abbrev, span, scope='global'):
        self.number_of_alleles = len(span)
        axes = axis_labels_inheritance(name)
        elements = element_labels_inheritance(abbrev, span)
        self.labels = {'axes': axes, 'elements': elements}
        self.info = "nuclear inheritance at %s locus with %d alleles" % (name, self.number_of_alleles)
        base.Weight.__init__(self, locus_name=name, locus_abbrev=abbrev, allele_range=span, _scope=scope)
    
    def _build(self):
        ary = nuclear_inheritance_at_single_locus(self.number_of_alleles)
        return ary

class NuclearInheritanceAtYLocus(base.Weight):
    _type = 'constant reproduction'
    
    def __init__(self, number_of_male_y_chromosomes, scope='global'):
        self.number_of_alleles = number_of_male_y_chromosomes + 1
        name = 'y-chromosome'
        abbrev = 'Y'
        span = range(self.number_of_alleles)
        axes = axis_labels_inheritance(name)
        elements = element_labels_inheritance(abbrev, span)
        self.labels = {'axes': axes, 'elements': elements}
        self.info = "nuclear inheritance at %s locus with %d alleles (first allele is 'Y0' and means 'female')" % (name, self.number_of_alleles)
        base.Weight.__init__(self, locus_name=name, locus_abbrev=abbrev, allele_range=span, _scope=scope)
    
    def _build(self):
        """1:1 sex ratio!"""
        ary = nuclear_inheritance_at_y_locus(self.number_of_alleles)
        return ary

class NuclearInheritanceAtTwoLoci(base.Weight):
    _type = 'constant reproduction'
    _parameter_names = ['recombination_rate']
    _info = 'nuclear inheritance at 2 loci'
    
    def __init__(self, recombination_rate, name1, abbrev1, span1, name2, abbrev2, span2, scope='global'):
        self.number_of_alleles1 = len(span1)
        self.number_of_alleles2 = len(span2)
        axes = axis_labels_inheritance(name1, name2)
        elements = element_labels_inheritance(abbrev1, span1, abbrev2, span2)
        self.labels = {'axes': axes, 'elements': elements}
        self.info = "inheritance at %s and %s loci with %d and %d alleles, respectively" % (name1, name2, self.number_of_alleles1, self.number_of_alleles2)
        base.Weight.__init__(self, locus_name1=name1, locus_abbrev1=abbrev1, allele_range1=span1, locus_name2=name2, locus_abbrev2=abbrev2, allele_range2=span2, recombination_rate=recombination_rate, _scope=scope)
        
    def _build(self):
        ary = nuclear_inheritance_at_two_loci(self.number_of_alleles1, self.number_of_alleles2, self.recombination_rate)
        return ary

class MatingPreference(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'preference_alleles']
    labels = dict( axes=['female preference', 'male trait'],
                   elements=[] )
    _info = 'female mating preference'
    """P1 females prefer T1 males, costly preference.
    mating preference strength:          0 <= a <= inf
    rejection probability of mating:     0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'P0','None',0., 'P1','T1',0.9, 'P2','T2',0.5, ...
        (preference allele, preferred trait, rejection probability / preference strength)
        possible modes: pr / a   (rejection probability / preference strength)
        """
        preference_alleles = {}
        n = int(len(args)/3.)        # number of preference alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, preferred_trait, pr = args[i*3:(i+1)*3]
                pr = float(pr)
                if pr == 1.:
                    a = np.inf
                else:
                    a = pr / (1-pr)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait}
        elif mode == 'a':
            for i in range(n):
                allele_name, preferred_trait, a = args[i*3:(i+1)*3]
                if a == np.inf:
                    pr = 1.
                else:
                    a = float(a)        # if an integer was provided, turn into float!
                    pr = a / (1+a)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait}
        self.preference_allele_names = sorted(preference_alleles.keys())
        self.number_of_preference_alleles = len(self.preference_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, preference_alleles=preference_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        trait_alleles = self.pop._get_axis_elements('trait')
        self.number_of_traits = len(trait_alleles)
        elements = [["f%s" % pref for pref in self.preference_allele_names]]
        elements.append( ["m%s" % trait for trait in trait_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_preference_alleles, self.number_of_traits)
        self._init_labels(self.labels)
        self.data = self._calc()
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2))
    
    def _calc(self):
        MP = np.ones(self._data_shape)
        pt = self.transition_probability
        for pref_allele in self.preference_allele_names:
            pdict = self.preference_alleles[pref_allele]
            pref_trait = pdict['preferred_trait']
            if pref_trait != 'None':
                x = self.pop.get_sum(pref_trait)
                i,j = self.pop.get_position(pref_allele)[1], self.pop.get_position(pref_trait)[1]
                pr = pdict['rejection_probability']
                R = 1. / (1 - pr*pt*(1-x))
                MP[i] *= (1-pr)*R
                MP[i,j] = R
        return np.nan_to_num(MP)           # replace NaN with zero (happens when pr=pt=1 and x=0)
    
class MatingPreferenceT(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'preference_alleles']
    labels = dict( axes=['female trait-preference', 'male trait'],
                   elements=[] )
    _info = 'female mating preference for male traits'
    """P1 females prefer T1 males, costly preference.
    mating preference strength:          0 <= a <= inf
    rejection probability of mating:     0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'P0','None',0., 'P1','T1',0.9, 'P2','T2',0.5, ...
        (preference allele, preferred trait, rejection probability / preference strength)
        possible modes: pr / a   (rejection probability / preference strength)
        """
        preference_alleles = {}
        n = int(len(args)/3.)        # number of preference alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, preferred_trait, pr = args[i*3:(i+1)*3]
                pr = float(pr)
                if pr == 1.:
                    a = np.inf
                else:
                    a = pr / (1-pr)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait}
        elif mode == 'a':
            for i in range(n):
                allele_name, preferred_trait, a = args[i*3:(i+1)*3]
                if a == np.inf:
                    pr = 1.
                else:
                    a = float(a)        # if an integer was provided, turn into float!
                    pr = a / (1+a)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait}
        self.preference_allele_names = sorted(preference_alleles.keys())
        self.number_of_preference_alleles = len(self.preference_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, preference_alleles=preference_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        trait_alleles = self.pop._get_axis_elements('trait')
        self.number_of_traits = len(trait_alleles)
        elements = [["f%s" % pref for pref in self.preference_allele_names]]
        elements.append( ["m%s" % trait for trait in trait_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_preference_alleles, self.number_of_traits)
        self._init_labels(self.labels)
        self.data = self._calc()
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2))
    
    def _calc(self):
        MP = np.ones(self._data_shape)
        pt = self.transition_probability
        for pref_allele in self.preference_allele_names:
            pdict = self.preference_alleles[pref_allele]
            pref_trait = pdict['preferred_trait']
            if pref_trait != 'None':
                x = self.pop.get_sum(pref_trait)
                i,j = self.pop.get_position(pref_allele)[1], self.pop.get_position(pref_trait)[1]
                pr = pdict['rejection_probability']
                R = 1. / (1 - pr*pt*(1-x))
                MP[i] *= (1-pr)*R
                MP[i,j] = R
        return np.nan_to_num(MP)           # replace NaN with zero (happens when pr=pt=1 and x=0)

class MatingPreferenceABT(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'preference_alleles']
    labels = dict( axes=['female preference', 'male background-A', 'male background-B', 'male trait'],
                   elements=[] )
    _info = 'female mating preference'
    """P1 females prefer T1 and A1B1 males, costly preference.
    mating preference strength:          0 <= a <= inf
    rejection probability of mating:     0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'P0','None','None',0., 'P1','T1','A1-B1',0.9, 'P2','T2','A2-B2',0.5, ...
        (preference allele, preferred trait, preferred background, rejection probability / preference strength)
        possible modes: pr / a   (rejection probability / preference strength)
        """
        preference_alleles = {}
        n = int(len(args)/3.)        # number of preference alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, preferred_trait, preferred_background, pr = args[i*4:(i+1)*4]
                pr = float(pr)
                if pr == 1.:
                    a = np.inf
                else:
                    a = pr / (1-pr)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait,
                                                   'preferred_background': preferred_background}
        elif mode == 'a':
            for i in range(n):
                allele_name, preferred_trait, preferred_background, a = args[i*4:(i+1)*4]
                if a == np.inf:
                    pr = 1.
                else:
                    a = float(a)        # if an integer was provided, turn into float!
                    pr = a / (1+a)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait,
                                                   'preferred_background': preferred_background}
        self.preference_allele_names = sorted(preference_alleles.keys())
        self.number_of_preference_alleles = len(self.preference_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, preference_alleles=preference_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        trait_alleles = self.pop._get_axis_elements('trait')
        A_alleles = self.pop._get_axis_elements('background-A')
        B_alleles = self.pop._get_axis_elements('background-B')
        self.number_of_traits = len(trait_alleles)
        self.number_of_A_alleles = len(A_alleles)
        self.number_of_B_alleles = len(B_alleles)
        elements = [["f%s" % pref for pref in self.preference_allele_names]]
        elements.append( ["m%s" % A for A in A_alleles] )
        elements.append( ["m%s" % B for B in B_alleles] )
        elements.append( ["m%s" % trait for trait in trait_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_preference_alleles, self.number_of_A_alleles, self.number_of_B_alleles, self.number_of_traits)
        self._init_labels(self.labels)
        self.data = self._calc()
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2))
    
    def _calc(self):
        MP = np.ones(self._data_shape)
        pt = self.transition_probability
        for pref_allele in self.preference_allele_names:
            pdict = self.preference_alleles[pref_allele]
            pref_trait = pdict['preferred_trait']
            pref_background = pdict['preferred_background']
            if pref_background != 'None':
                pref_Y,pref_A = pref_background.split('-')
            if pref_trait != 'None' or pref_background != 'None':
                xpref = self.pop.get_sum("%s-%s-%s" % (pref_A, pref_B, pref_trait))
                xtraitpref = self.pop.get_sum(pref_trait)
                xbackpref = self.pop.get_sum("%s-%s" % (pref_A, pref_B))
                y1 = xtraitpref + xbackpref -2*xpref
                y2 = 1 - y1 - xpref
                i,j,k,l = self.pop.get_position(pref_allele)[1], self.pop.get_position(pref_A)[1], self.pop.get_position(pref_B)[1], self.pop.get_position(pref_trait)[1]
                pr = pdict['rejection_probability']
                R = 1. / (1 - pr*pt*( y1+(2-pr)*y2) )
                MP[i]       *= (np.power(1-pr,2) * R)
                MP[i,j,k,:] /= (1-pr)
                MP[i,:,:,l] /= (1-pr)
        return np.nan_to_num(MP)           # replace NaN with zero (happens when pr=pt=1 and x=0)

class MatingPreferenceAB(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'preference_alleles']
    labels = dict( axes=['female preference', 'male background-A', 'male background-B'],
                   elements=[] )
    _info = 'female mating preference for background loci'
    """P1 females prefer A1B1 males, costly preference.
    mating preference strength:          0 <= a <= inf
    rejection probability of mating:     0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'P0','None',0., 'P1','A1-B1',0.9, 'P2','A2-B2',0.5, ...
        (preference allele, preferred background, rejection probability / preference strength)
        possible modes: pr / a   (rejection probability / preference strength)
        """
        preference_alleles = {}
        n = int(len(args)/3.)        # number of preference alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, preferred_background, pr = args[i*3:(i+1)*3]
                pr = float(pr)
                if pr == 1.:
                    a = np.inf
                else:
                    a = pr / (1-pr)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_background': preferred_background}
        elif mode == 'a':
            for i in range(n):
                allele_name, preferred_background, a = args[i*3:(i+1)*3]
                if a == np.inf:
                    pr = 1.
                else:
                    a = float(a)        # if an integer was provided, turn into float!
                    pr = a / (1+a)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_background': preferred_background}
        self.preference_allele_names = sorted(preference_alleles.keys())
        self.number_of_preference_alleles = len(self.preference_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, preference_alleles=preference_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        A_alleles = self.pop._get_axis_elements('background-A')
        B_alleles = self.pop._get_axis_elements('background-B')
        self.number_of_A_alleles = len(A_alleles)
        self.number_of_B_alleles = len(B_alleles)
        elements = [["f%s" % pref for pref in self.preference_allele_names]]
        elements.append( ["m%s" % A for A in A_alleles] )
        elements.append( ["m%s" % B for B in B_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_preference_alleles, self.number_of_A_alleles, self.number_of_B_alleles)
        self._init_labels(self.labels)
        self.data = self._calc()        
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2))
    
    def _calc(self):
        MP = np.ones(self._data_shape)
        pt = self.transition_probability
        for pref_allele in self.preference_allele_names:
            pdict = self.preference_alleles[pref_allele]
            pref_background = pdict['preferred_background']
            if pref_background != 'None':
                pref_A,pref_B = pref_background.split('-')
                xbackpref = self.pop.get_sum(pref_background)
                i,j,k = self.pop.get_position(pref_allele)[1], self.pop.get_position(pref_A)[1], self.pop.get_position(pref_B)[1]
                pr = pdict['rejection_probability']
                R = 1. / (1 - pr*pt*(1-xbackpref) )                
                MP[i]     *= ((1-pr)*R)
                MP[i,j,k]  = R
        return np.nan_to_num(MP)           # replace NaN with zero (happens when pr=pt=1 and x=0)

class MatingPreferenceYA(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'preference_alleles']
    labels = dict( axes=['female background-preference', 'male y-chromosome', 'male background-A'],
                   elements=[] )
    _info = 'female mating preference for background loci'
    """P1 females prefer Y1A1 males, costly preference.
    mating preference strength:          0 <= a <= inf
    rejection probability of mating:     0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'P0','None',0., 'P1','Y1-A1',0.9, 'P2','Y2-A2',0.5, ...
        (preference allele, preferred background, rejection probability / preference strength)
        possible modes: pr / a   (rejection probability / preference strength)
        """
        preference_alleles = {}
        n = int(len(args)/3.)        # number of preference alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, preferred_background, pr = args[i*3:(i+1)*3]
                pr = float(pr)
                if pr == 1.:
                    a = np.inf
                else:
                    a = pr / (1-pr)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_background': preferred_background}
        elif mode == 'a':
            for i in range(n):
                allele_name, preferred_background, a = args[i*3:(i+1)*3]
                if a == np.inf:
                    pr = 1.
                else:
                    a = float(a)        # if an integer was provided, turn into float!
                    pr = a / (1+a)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_background': preferred_background}
        self.preference_allele_names = sorted(preference_alleles.keys())
        self.number_of_preference_alleles = len(self.preference_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, preference_alleles=preference_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        Y_alleles = self.pop._get_axis_elements('y-chromosome')   #[1:]    # first element denotes females
        A_alleles = self.pop._get_axis_elements('background-A')
        self.number_of_Y_alleles = len(Y_alleles)
        self.number_of_A_alleles = len(A_alleles)
        elements = [["f%s" % pref for pref in self.preference_allele_names]]
        elements.append( ["m%s" % Y for Y in Y_alleles] )
        elements.append( ["m%s" % A for A in A_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_preference_alleles, self.number_of_Y_alleles, self.number_of_A_alleles)
        self._init_labels(self.labels)
        self.data = self._calc()        
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2))
    
    def _calc(self):
        MP = np.ones(self._data_shape)
        MP[:,0,:] = 0.          # no matings between females
        pt = self.transition_probability
        for pref_allele in self.preference_allele_names:
            pdict = self.preference_alleles[pref_allele]
            pref_background = pdict['preferred_background']
            if pref_background != 'None':
                pref_Y,pref_A = pref_background.split('-')
                xbackpref = 2 * self.pop.get_sum(pref_background)   # factor 2 because half of the population are Y0 (females)
                i,j,k = self.pop.get_position(pref_allele)[1], self.pop.get_position(pref_Y)[1], self.pop.get_position(pref_A)[1]
                pr = pdict['rejection_probability']
                R = 1. / (1 - pr*pt*(1-xbackpref) )                
                MP[i]     *= (1-pr)*R
                MP[i,j,k]  = R
        return np.nan_to_num(MP)           # replace NaN with zero (happens when pr=pt=1 and x=0)

class MatingPreferenceYAT(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'preference_alleles']
    labels = dict( axes=['female preference', 'male y-chromosome', 'male background-A', 'male trait'],
                   elements=[] )
    _info = 'female mating preference'
    """P1 females prefer T1 and Y1A1 males, costly preference.
    mating preference strength:          0 <= a <= inf
    rejection probability of mating:     0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'P0','None','None',0., 'P1','T1','Y1-A1',0.9, 'P2','T2','Y2-A2',0.5, ...
        (preference allele, preferred trait, preferred background, rejection probability / preference strength)
        possible modes: pr / a   (rejection probability / preference strength)
        """
        preference_alleles = {}
        n = int(len(args)/3.)        # number of preference alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, preferred_trait, preferred_background, pr = args[i*4:(i+1)*4]
                pr = float(pr)
                if pr == 1.:
                    a = np.inf
                else:
                    a = pr / (1-pr)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait,
                                                   'preferred_background': preferred_background}
        elif mode == 'a':
            for i in range(n):
                allele_name, preferred_trait, preferred_background, a = args[i*4:(i+1)*4]
                if a == np.inf:
                    pr = 1.
                else:
                    a = float(a)        # if an integer was provided, turn into float!
                    pr = a / (1+a)
                preference_alleles[allele_name] = {'preference_strength': a,
                                                   'rejection_probability': pr,
                                                   'preferred_trait': preferred_trait,
                                                   'preferred_background': preferred_background}
        self.preference_allele_names = sorted(preference_alleles.keys())
        self.number_of_preference_alleles = len(self.preference_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, preference_alleles=preference_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        trait_alleles = self.pop._get_axis_elements('trait')
        Y_alleles = self.pop._get_axis_elements('y-chromosome')   #[1:]    # first element denotes females
        A_alleles = self.pop._get_axis_elements('background-A')
        self.number_of_traits = len(trait_alleles)
        self.number_of_Y_alleles = len(Y_alleles)
        self.number_of_A_alleles = len(A_alleles)
        elements = [["f%s" % pref for pref in self.preference_allele_names]]
        elements.append( ["m%s" % Y for Y in Y_alleles] )
        elements.append( ["m%s" % A for A in A_alleles] )
        elements.append( ["m%s" % trait for trait in trait_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_preference_alleles, self.number_of_Y_alleles, self.number_of_A_alleles, self.number_of_traits)
        self._init_labels(self.labels)
        self.data = self._calc()        
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2))
    
    def _calc(self):
        MP = np.ones(self._data_shape)
        MP[:,0,:,:] = 0.          # no matings between females
        pt = self.transition_probability
        for pref_allele in self.preference_allele_names:
            pdict = self.preference_alleles[pref_allele]
            pref_trait = pdict['preferred_trait']
            pref_background = pdict['preferred_background']
            if pref_background != 'None':
                pref_Y,pref_A = pref_background.split('-')
            if pref_trait != 'None' or pref_background != 'None':
                xpref = 2 * self.pop.get_sum("%s-%s-%s" % (pref_Y, pref_A, pref_trait))  # factor 2 because half of the population are Y0 (females)
                xtraitpref = self.pop.get_sum(pref_trait)
                xbackpref = 2 * self.pop.get_sum("%s-%s" % (pref_Y, pref_A))   # factor 2 because half of the population are Y0 (females)
                y1 = xtraitpref + xbackpref -2*xpref
                y2 = 1 - y1 - xpref
                i,j,k,l = self.pop.get_position(pref_allele)[1], self.pop.get_position(pref_Y)[1], self.pop.get_position(pref_A)[1], self.pop.get_position(pref_trait)[1]
                pr = pdict['rejection_probability']
                R = 1. / (1 - pr*pt*( y1+(2-pr)*y2) )
#                MP[i]       *= (np.power(1-pr,2) * R)
#                MP[i,j,k,:] /= (1-pr)
#                MP[i,:,:,l] /= (1-pr)
                
                MP[i]       *= (np.power(1-pr,2) * R)
                MP[i,:,:,l]  = (1-pr)*R
                MP[i,j,k,:]  = (1-pr)*R
                MP[i,j,k,l]  = R
        return np.nan_to_num(MP)           # replace NaN with zero (happens when pr=pt=1 and x=0)

class AssortativeMating(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'assortative_alleles']
    labels = dict( axes=['female assortative', 'female trait', 'male trait'],
                   elements=[] )
    _info = 'female assortative mating (trait)'
    """Q1 females prefer males with whom they share the same allele at the trait locus, costly preference.
    assortative mating strength: 0 <= c <= inf
    mating probability:          0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'Q0',0., 'Q1',0.9, 'Q2',0.5, ...
        (assortative mating allele, rejection probability / assortative mating strength)
        possible modes: pr / c (rejection probability / assortative mating strength)
        """
        assortative_alleles = {}
        n = int(len(args)/2.)        # number of assortative mating alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, pr = args[i*2:(i+1)*2]
                pr = float(pr)
                if pr == 1.:
                    c = np.inf
                else:
                    c = pr / (1-pr)
                assortative_alleles[allele_name] = {'assortative_strength': c,
                                                    'rejection_probability': pr}
        elif mode == 'c':
            for i in range(n):
                allele_name, c = args[i*2:(i+1)*2]
                if c == np.inf:
                    pr = 1.
                else:
                    c = float(c)        # if an integer was provided, turn into float!
                    pr = c / (1+c)
                preference_alleles[allele_name] = {'assortative_strength': c,
                                                   'rejection_probability': pr}
        self.assortative_allele_names = sorted(assortative_alleles.keys())
        self.number_of_assortative_alleles = len(self.assortative_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, assortative_alleles=assortative_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        self.trait_alleles = self.pop._get_axis_elements('trait')
        self.number_of_traits = len(self.trait_alleles)
        elements = [["f%s" % ass for ass in self.assortative_allele_names]]    # female assortative allele
        elements.append( ["f%s" % trait for trait in self.trait_alleles] )          # female trait
        elements.append( ["m%s" % trait for trait in self.trait_alleles] )          # male trait
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_assortative_alleles, self.number_of_traits, self.number_of_traits)
        self._init_labels(self.labels)
        self.data = self._calc()
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2,2))
    
    def _calc(self):
        AM = np.ones(self._data_shape)
        pt = self.transition_probability
        for ass_allele in self.assortative_allele_names:
            adict = self.assortative_alleles[ass_allele]
            pr = adict['rejection_probability']
            if pr != 0.:
                for ftrait in self.trait_alleles:
#                    pdb.set_trace()
                    x = self.pop.get_sum(ftrait)
                    R = 1. / (1 - pr*pt*(1-x))
                    i,j = self.pop.get_position(ass_allele)[1], \
                          self.pop.get_position(ftrait)[1]
                    AM[i,j] *= R
                    for mtrait in self.trait_alleles:
                        k = self.pop.get_position(mtrait)[1]
                        if k != j:
                            AM[i,j,k] *= 1-pr
        return np.nan_to_num(AM)

class AssortativeMatingT(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'assortative_alleles']
    labels = dict( axes=['female trait-assortative', 'female trait', 'male trait'],
                   elements=[] )
    _info = 'female assortative mating (trait)'
    """Q1 females prefer males with whom they share the same allele at the trait locus, costly preference.
    assortative mating strength: 0 <= c <= inf
    mating probability:          0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'TA0',0., 'TA1',0.9, 'TA2',0.5, ...
        (assortative mating allele, rejection probability / assortative mating strength)
        possible modes: pr / c (rejection probability / assortative mating strength)
        """
        assortative_alleles = {}
        n = int(len(args)/2.)        # number of assortative mating alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, pr = args[i*2:(i+1)*2]
                pr = float(pr)
                if pr == 1.:
                    c = np.inf
                else:
                    c = pr / (1-pr)
                assortative_alleles[allele_name] = {'assortative_strength': c,
                                                    'rejection_probability': pr}
        elif mode == 'c':
            for i in range(n):
                allele_name, c = args[i*2:(i+1)*2]
                if c == np.inf:
                    pr = 1.
                else:
                    c = float(c)        # if an integer was provided, turn into float!
                    pr = c / (1+c)
                preference_alleles[allele_name] = {'assortative_strength': c,
                                                   'rejection_probability': pr}
        self.assortative_allele_names = sorted(assortative_alleles.keys())
        self.number_of_assortative_alleles = len(self.assortative_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, assortative_alleles=assortative_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        self.trait_alleles = self.pop._get_axis_elements('trait')
        self.number_of_traits = len(self.trait_alleles)
        elements = [["f%s" % ass for ass in self.assortative_allele_names]]    # female assortative allele
        elements.append( ["f%s" % trait for trait in self.trait_alleles] )          # female trait
        elements.append( ["m%s" % trait for trait in self.trait_alleles] )          # male trait
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_assortative_alleles, self.number_of_traits, self.number_of_traits)
        self._init_labels(self.labels)
        self.data = self._calc()
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2,2))
    
    def _calc(self):
        AM = np.ones(self._data_shape)
        pt = self.transition_probability
        for ass_allele in self.assortative_allele_names:
            adict = self.assortative_alleles[ass_allele]
            pr = adict['rejection_probability']
            if pr != 0.:
                for ftrait in self.trait_alleles:
#                    pdb.set_trace()
                    x = self.pop.get_sum(ftrait)
                    R = 1. / (1 - pr*pt*(1-x))
                    i,j = self.pop.get_position(ass_allele)[1], \
                          self.pop.get_position(ftrait)[1]
                    AM[i,j]   *= ((1-pr)*R)
                    AM[i,j,j]  = R
#                    for mtrait in self.trait_alleles:
#                        k = self.pop.get_position(mtrait)[1]
#                        if k != j:
#                            AM[i,j,k] *= 1-pr
        return np.nan_to_num(AM)

class AssortativeMatingAB(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'assortative_alleles']
    labels = dict( axes=['female background-assortative', 'female background-A', 'female background-B', 'male background-A', 'male background-B'],
                   elements=[] )
    _info = 'female assortative mating (background loci)'
    """Q1 females prefer males with whom they share the same alleles at the background loci, costly preference.
    assortative mating strength: 0 <= c <= inf
    mating probability:          0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'Q0',0., 'Q1',0.9, 'Q2',0.5, ...
        (assortative mating allele, rejection probability / assortative mating strength)
        possible modes: pr / c (rejection probability / assortative mating strength)
        """
        assortative_alleles = {}
        n = int(len(args)/2.)        # number of assortative mating alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, pr = args[i*2:(i+1)*2]
                pr = float(pr)
                if pr == 1.:
                    c = np.inf
                else:
                    c = pr / (1-pr)
                assortative_alleles[allele_name] = {'assortative_strength': c,
                                                    'rejection_probability': pr}
        elif mode == 'c':
            for i in range(n):
                allele_name, c = args[i*2:(i+1)*2]
                if c == np.inf:
                    pr = 1.
                else:
                    c = float(c)        # if an integer was provided, turn into float!
                    pr = c / (1+c)
                preference_alleles[allele_name] = {'assortative_strength': c,
                                                   'rejection_probability': pr}
        self.assortative_allele_names = sorted(assortative_alleles.keys())
        self.number_of_assortative_alleles = len(self.assortative_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, assortative_alleles=assortative_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        self.A_alleles = self.pop._get_axis_elements('background-A')
        self.B_alleles = self.pop._get_axis_elements('background-B')
        self.number_of_A_alleles = len(self.A_alleles)
        self.number_of_B_alleles = len(self.B_alleles)
        elements = [["f%s" % pref for pref in self.assortative_allele_names]]
        elements.append( ["f%s" % A for A in self.A_alleles] )
        elements.append( ["f%s" % B for B in self.B_alleles] )
        elements.append( ["m%s" % A for A in self.A_alleles] )
        elements.append( ["m%s" % B for B in self.B_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_assortative_alleles, self.number_of_A_alleles, self.number_of_B_alleles, self.number_of_A_alleles, self.number_of_B_alleles)
        self._init_labels(self.labels)
        self.data = self._calc()
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2,2))
    
    def _calc(self):
        AM = np.ones(self._data_shape)
        pt = self.transition_probability
        for ass_allele in self.assortative_allele_names:
            adict = self.assortative_alleles[ass_allele]
            pr = adict['rejection_probability']
            if pr != 0.:
                for A in self.A_alleles:
                    for B in self.B_alleles:
    #                    pdb.set_trace()
                        x = self.pop.get_sum("%s-%s" % (A,B))
                        R = 1. / (1 - pr*pt*(1-x))
                        i,j,k = self.pop.get_position(ass_allele)[1], \
                                self.pop.get_position(A)[1], \
                                self.pop.get_position(B)[1]
                        AM[i,j,k]     *= ((1-pr)*R)
                        AM[i,j,k,j,k]  = R
        return np.nan_to_num(AM)

#    def _calc(self):
#        AM = np.ones(self._data_shape)
#        pt = self.transition_probability
#        for ass_allele in self.assortative_allele_names:
#            adict = self.assortative_alleles[ass_allele]
#            pr = adict['rejection_probability']
#            if pr != 0.:
#                for ftrait in self.trait_alleles:
#                    x = self.pop.get_sum(ftrait)
#                    R = 1. / (1 - pr*pt*(1-x))
#                    i,j = self.pop.get_position(ass_allele)[1], \
#                          self.pop.get_position(ftrait)[1]
#                    AM[i,j] *= (1-pr)*R
#                    # if male trait == female trait:
#                    AM[i,j,j] /= (1-pr)
#        return AM    

class AssortativeMatingAB2(base.Weight):
    _type = 'dynamic reproduction'
    _parameter_names = ['transition_probability', 'assortative_alleles']
    labels = dict( axes=['female background-assortative', 'female locus-A', 'female locus-B', 'male locus-A', 'male locus-B'],
                   elements=[] )
    _info = 'female assortative mating (background loci)'
    """QAB1 females prefer males with whom they share the same alleles at the background loci, costly preference.
    assortative mating strength: 0 <= c <= inf
    mating probability:          0 <= p <= 1
    """
    
    def __init__(self, transition_probability, mode, *args):
        """Usage: args = 'QAB0',0., 'QAB1',0.9, 'QAB2',0.5, ...
        (assortative mating allele, rejection probability / assortative mating strength)
        possible modes: pr / c (rejection probability / assortative mating strength)
        """
        assortative_alleles = {}
        n = int(len(args)/2.)        # number of assortative mating alleles
        if mode == 'pr':
            for i in range(n):
                allele_name, pr = args[i*2:(i+1)*2]
                pr = float(pr)
                if pr == 1.:
                    c = np.inf
                else:
                    c = pr / (1-pr)
                assortative_alleles[allele_name] = {'assortative_strength': c,
                                                    'rejection_probability': pr}
        elif mode == 'c':
            for i in range(n):
                allele_name, c = args[i*2:(i+1)*2]
                if c == np.inf:
                    pr = 1.
                else:
                    c = float(c)        # if an integer was provided, turn into float!
                    pr = c / (1+c)
                preference_alleles[allele_name] = {'assortative_strength': c,
                                                   'rejection_probability': pr}
        self.assortative_allele_names = sorted(assortative_alleles.keys())
        self.number_of_assortative_alleles = len(self.assortative_allele_names)
        base.Weight.__init__(self, transition_probability=transition_probability, assortative_alleles=assortative_alleles, _scope='population')
     
    def set(self):
        print "parameters not settable!"
    
    def _init_after_assignment(self):
        self.A_alleles = self.pop._get_axis_elements('locus-A')
        self.B_alleles = self.pop._get_axis_elements('locus-B')
        self.number_of_A_alleles = len(self.A_alleles)
        self.number_of_B_alleles = len(self.B_alleles)
        elements = [["f%s" % pref for pref in self.assortative_allele_names]]
        elements.append( ["f%s" % A for A in self.A_alleles] )
        elements.append( ["f%s" % B for B in self.B_alleles] )
        elements.append( ["m%s" % A for A in self.A_alleles] )
        elements.append( ["m%s" % B for B in self.B_alleles] )
        self.labels['elements'] = elements
        self._data_shape = (self.number_of_assortative_alleles, self.number_of_A_alleles, self.number_of_B_alleles, self.number_of_A_alleles, self.number_of_B_alleles)
        self._init_labels(self.labels)
        self.data = self._calc()
        self._ndim = self.data.ndim
        
    def _build(self):
        """dummy method"""
        return np.zeros((2,2,2))
    
    def _calc(self):
        AM = np.ones(self._data_shape)
        pt = self.transition_probability
        for ass_allele in self.assortative_allele_names:
            adict = self.assortative_alleles[ass_allele]
            pr = adict['rejection_probability']
            if pr != 0.:
                for A in self.A_alleles:
                    for B in self.B_alleles:
    #                    pdb.set_trace()
                        x = self.pop.get_sum("%s-%s" % (A,B))
                        R = 1. / (1 - pr*pt*(1-x))
                        i,j,k = self.pop.get_position(ass_allele)[1], \
                                self.pop.get_position(A)[1], \
                                self.pop.get_position(B)[1]
                        AM[i,j,k]     *= ((1-pr)*R)
                        AM[i,j,k,j,k]  = R
        return np.nan_to_num(AM)

class ViabilitySelection(base.Weight):
    """Selection coefficients are ADDED to average viability of 1!"""
    _type = 'constant viability selection'
    _parameter_names = ['selection_coefficients']
    labels = { 'axes': ['trait'], \
               'elements': [] }
    name = 'VS'
    _info = 'viability selection'
               
    def __init__(self, selection_coefficients, scope='population'):
        self.number_of_alleles = len(selection_coefficients)
        self.info = "viability selection at trait locus with %d alleles" % self.number_of_alleles
        base.Weight.__init__(self, selection_coefficients=selection_coefficients, _scope=scope)
    
    def _init_after_assignment(self):
        elements = self.pop._get_axis_elements('trait')
        self.labels['elements'] = [elements]
        self._init_labels(self.labels)
    
    def _build(self):
        ary = np.array(self.selection_coefficients) + 1.
        return ary

class ViabilitySelectionAtCustomLocus(base.Weight):
    """Selection coefficients are ADDED to average viability of 1!"""
    _type = 'constant viability selection'
    _parameter_names = ['locus','selection_coefficients']
    name = 'VSC'
               
    def __init__(self, locus, selection_coefficients, scope='population'):
        self.number_of_alleles = len(selection_coefficients)
        self.info = "viability selection at %s locus" % locus
        base.Weight.__init__(self, locus=locus, selection_coefficients=selection_coefficients, _scope=scope)
    
    def _init_after_assignment(self):
        """This method is called after assignment to a populationp."""
        self.labels = { 'axes': [self.locus], \
                        'elements': [self.pop._get_axis_elements(self.locus)] }
        self._init_labels(self.labels)
    
    def _build(self):
        ary = np.array(self.selection_coefficients) + 1.
        return ary

class ViabilitySelectionAt2CustomLoci(base.Weight):
    """Selection coefficients are ADDED to average viability of 1!"""
    _type = 'constant viability selection'
    _parameter_names = ['locus1', 'locus2', 'selection_coefficients']
    name = 'VSC'
               
    def __init__(self, locus1, locus2, selection_coefficients, scope='population'):
        self.number_of_alleles = len(selection_coefficients)
        self.info = "viability selection at %s and %s loci" % (locus1, locus2)
        base.Weight.__init__(self, locus1=locus1, locus2=locus2, selection_coefficients=selection_coefficients, _scope=scope)
    
    def _init_after_assignment(self):
        """This method is called after assignment to a populationp."""
        self.labels = { 'axes': [self.locus1, self.locus2], \
                        'elements': [self.pop._get_axis_elements(self.locus1), self.pop._get_axis_elements(self.locus2)] }
        self._init_labels(self.labels)
#        print self.labels
#        print self.__dict__
    
    def _build(self):
        ary = np.array(self.selection_coefficients) + 1.
        return ary

class ViabilitySelectionAt3CustomLoci(base.Weight):
    """Selection coefficients are ADDED to average viability of 1!"""
    _type = 'constant viability selection'
    _parameter_names = ['locus1', 'locus2', 'locus3', 'selection_coefficients']
    name = 'VSC'
               
    def __init__(self, locus1, locus2, locus3, selection_coefficients, scope='population'):
        self.number_of_alleles = len(selection_coefficients)
        self.info = "viability selection at 3 loci: %s, %s, and %s" % (locus1,locus2,locus3)
        base.Weight.__init__(self, locus1=locus1, locus2=locus2, locus3=locus3, selection_coefficients=selection_coefficients, _scope=scope)
    
    def _init_after_assignment(self):
        """This method is called after assignment to a populationp."""
        self.labels = { 'axes': [self.locus1, self.locus2, self.locus3], \
                        'elements': [self.pop._get_axis_elements(self.locus1), self.pop._get_axis_elements(self.locus2), self.pop._get_axis_elements(self.locus3)] }
        self._init_labels(self.labels)
    
    def _build(self):
        ary = np.array(self.selection_coefficients) + 1.
        return ary
        
class ViabilitySelectionMales(base.Weight):
    """Attention: Only use with explicit sex populations!"""
    _type = 'constant viability selection'
    _parameter_names = ['selection_coefficients']
    labels = { 'axes': ['sex', 'trait'], \
               'elements': [] }
    name = 'VSM'
               
    def __init__(self, selection_coefficients, scope='population'):
        self.number_of_alleles = len(selection_coefficients)
        self.info = "viability selection at male trait locus with %d alleles" % self.number_of_alleles
        base.Weight.__init__(self, selection_coefficients=selection_coefficients, _scope=scope)
    
    def _init_after_assignment(self):
        self.labels['elements'] = [['f','m'],self.pop._get_axis_elements('trait')]
        self._init_labels(self.labels)
        # switch population viability mode to 'sex dependent':
        self.pop._viability_selection_mode = 'sex dependent'
    
    def _build(self):
        l = len(self.selection_coefficients)
        ary = np.ones((l,l))
        ary[1] = np.array(self.selection_coefficients) + 1.
        return ary

class NuclearBackgroundInheritance(base.Weight):
    _type = 'constant reproduction'
    _info = 'nuclear background inheritance'
    labels = { 'axes': ['female background', 'male background', 'offspring background'], \
               'elements': [['fNB1','fNB2','fNBH'], \
                            ['mNB1','mNB2','mNBH'], \
                            ['oNB1','oNB2','oNBH']] }
    
    def __init__(self, scope='global'):
        base.Weight.__init__(self, _scope=scope)
    
    def _build(self):
        """Build weight for the inheritance of nuclear background (a.k.a species).

                       male
            |  A         B         H
        ----+------------------------------------
          A |  A         H         A/2,H/2
female    B |  H         B         B/2,H/2
          H |  A/2,H/2   B/2,H/2   A/4,B/4,H/2
"""
        ary = np.zeros( (3,3,3), float )
        ary[0,0,0] = ary[1,1,1] = ary[0,1,2] = ary[1,0,2] = 1.
        ary[0,2,0] = ary[0,2,2] = ary[2,0,0] = ary[2,0,2] = 0.5
        ary[1,2,1] = ary[1,2,2] = ary[2,1,1] = ary[2,1,2] = 0.5
        ary[2,2,0] = ary[2,2,1] = 0.25
        ary[2,2,2] = 0.5
        return ary

class HybridMaleSterility(base.Weight):
    _type = 'constant reproduction'
    _info = 'hybrid male sterility (1 autosomal locus)'
    _parameter_names = ['sterility_coefficient']
    labels = { 'axes': ['male background'], \
               'elements': [['mNB1','mNB2','mNBH']] }
    
    def __init__(self, sterility_coefficient, scope='global'):
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, _scope=scope)
    
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [1, 1, 1-h], float )
        return ary

class HybridMaleSterility2Loci(base.Weight):
    """Ancestral population:            A0-B0
       after separation (4 mutations!): A1-B1 in left population
                                        A2-B2 in right population
                          incompatible: A1-B2 and A2-B1
    """
    _type = 'constant reproduction'
    _info = 'hybrid male sterility (2 autosomal loci)'
    _parameter_names = ['sterility_coefficient']
    labels = { 'axes': ['male background-A', 'male background-B'], \
               'elements': [['mA1', 'mA2'], \
                            ['mB1', 'mB2']] }
    
    def __init__(self, sterility_coefficient, scope='global'):
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, _scope=scope)
    
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary

class HybridMaleSterility2BackgoundLoci(base.Weight):
    """Ancestral population:            A0-B0
       after separation (4 mutations!): A1-B1 in left population
                                        A2-B2 in right population
                          incompatible: A1-B2 and A2-B1
    """
    _type = 'constant reproduction'
    _info = 'hybrid male sterility (2 autosomal loci)'
    _parameter_names = ['sterility_coefficient', 'locus1', 'locus2']
    
    def __init__(self, sterility_coefficient, locus1='A', locus2='B', scope='global'):
        labels = { 'axes': ["male background-%s" % locus1, "male background-%s" % locus2], \
                   'elements': [["m%s1" % locus1, "m%s2" % locus1], \
                                ["m%s1" % locus2, "m%s2" % locus2]] }
        self.labels = labels
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, locus1=locus1, locus2=locus2, _scope=scope)
            
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary

class HybridMaleSterility2CustomLoci(base.Weight):
    """Ancestral population:            A0-B0
       after separation (4 mutations!): A1-B1 in left population
                                        A2-B2 in right population
                          incompatible: A1-B2 and A2-B1
    """
    _type = 'constant reproduction'
    _info = 'hybrid male sterility (2 autosomal loci)'
    _parameter_names = ['sterility_coefficient', 'locus1', 'locus2']
    
    def __init__(self, sterility_coefficient, locus1='A', locus2='B', scope='global'):
        labels = { 'axes': ["male locus-%s" % locus1, "male locus-%s" % locus2], \
                   'elements': [["m%s1" % locus1, "m%s2" % locus1], \
                                ["m%s1" % locus2, "m%s2" % locus2]] }
        self.labels = labels
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, locus1=locus1, locus2=locus2, _scope=scope)
            
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary

class HybridMaleSterility2CustomLoci4Alleles(base.Weight):
    """Ancestral population:                        A0-B0
    
                                        left                   right
       after separation:                A1-B1         |        A4-B4
       
                                        left    hybrid zone    right
       towards secondary contact:       A1-B1 | A2-B2  A3-B3 | A4-B4
       
       compatible                       group I:  indexes 1/2
                                        group II: indexes 3/4
       incompatible                     combinations of groups I/II
                                        
    """
    _type = 'constant reproduction'
    _info = 'hybrid male sterility (2 autosomal loci with 4 alleles each)'
    _parameter_names = ['sterility_coefficient', 'locus1', 'locus2']
    
    def __init__(self, sterility_coefficient, locus1=('locus-A','A'), locus2=('locus-B','B'), scope='global'):
        abbreviations = [locus1[1], locus2[1]]
        labels = { 'axes': ["male %s" % locus1[0], "male %s" % locus2[0]], \
                   'elements': [["m%s%d" % (abbrev,i) for i in range(1,5)] for abbrev in abbreviations] }
        self.labels = labels
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, locus1=locus1, locus2=locus2, _scope=scope)
            
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [[1,   1,   1-h, 1-h], \
                        [1,   1,   1-h, 1-h], \
                        [1-h, 1-h, 1,   1  ],\
                        [1-h, 1-h, 1,   1  ]], float )
        return ary

class HybridMaleSterilityFasterMale(base.Weight):
    """Ancestral population: A-Y in males, A-Y0 in females
       after separation: A1-Y1 / A1-Y0 in left population
                         A2-Y2 / A2-Y0 in right population
                         incompatible: A1-Y2 and A2-Y1
    """
    _type = 'constant reproduction'
    _info = 'hybrid male sterility (faster male)'
    _parameter_names = ['sterility_coefficient']
    labels = { 'axes': ['male y-chromosome', 'male background-A'], \
               'elements': [['mY0', 'mY1', 'mY2'], \
                            ['mA1', 'mA2']] }
    
    def __init__(self, sterility_coefficient, scope='global'):
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, _scope=scope)
    
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [[1, 1], \
                        [1, 1-h], \
                        [1-h, 1]], float )
        return ary

class HybridFemaleSterility2Loci(base.Weight):
    """Ancestral population: A2-B1
       after separation: A1-B1 in left population
                         A2-B2 in right population
                         incompatible: A1-B2 and A2-B1
    """
    _type = 'constant reproduction'
    _info = 'hybrid female sterility'
    _parameter_names = ['sterility_coefficient']
    labels = { 'axes': ['female background-A', 'female background-B'], \
               'elements': [['fA1', 'fA2'], \
                            ['fB1', 'fB2']] }
    
    def __init__(self, sterility_coefficient, scope='global'):
        base.Weight.__init__(self, sterility_coefficient=sterility_coefficient, _scope=scope)
    
    def _build(self):
        h = self.sterility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary

class NuclearIncompatibility2Loci(base.Weight):
    """Ancestral population: A2-B1
       after separation: A1-B1 in left population
                         A2-B2 in right population
                         incompatible: A1-B2 and A2-B1
       Incompatibilities take effect in adult phases, i.e., they are a form
       of ecological inviability.
    """
    _type = 'constant viability selection'
    _info = 'nuclear incompatibility'
    _parameter_names = ['incompatibility_coefficient']
    labels = { 'axes': ['background-A', 'background-B'], \
               'elements': [['A1', 'A2'], \
                            ['B1', 'B2']] }
    
    def __init__(self, incompatibility_coefficient, scope='global'):
        base.Weight.__init__(self, incompatibility_coefficient=incompatibility_coefficient, _scope=scope)
    
    def _build(self):
        h = self.incompatibility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary
    
class HybridInviability2Loci(base.Weight):
    """Ancestral population: A2-B1
       after separation: A1-B1 in left population
                         A2-B2 in right population
                         incompatible: A1-B2 and A2-B1
       Incompatibilities occur in embryonic stages to the effect that only a fraction h 
       of pregeny with hybrid background are born at all.
    """
    _type = 'constant reproduction'
    _info = 'nuclear incompatibility'
    _parameter_names = ['incompatibility_coefficient']
    labels = { 'axes': ['offspring background-A', 'offspring background-B'], \
               'elements': [['oA1', 'oA2'], \
                            ['oB1', 'oB2']] }
    
    def __init__(self, incompatibility_coefficient, scope='global'):
        base.Weight.__init__(self, incompatibility_coefficient=incompatibility_coefficient, _scope=scope)
    
    def _build(self):
        h = self.incompatibility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary

class HybridInviability2LociAt(base.Weight):
    """Ancestral population: A2-B1
       after separation: A1-B1 in left population
                         A2-B2 in right population
                         incompatible: A1-B2 and A2-B1
       Incompatibilities occur in embryonic stages to the effect that only a fraction h 
       of pregeny with hybrid background are born at all.
    """
    _type = 'constant reproduction'
    _info = 'nuclear incompatibility'
    _parameter_names = ['incompatibility_coefficient']
    labels = { 'axes': [], 'elements': [] }
    
    def __init__(self, locus1, locus2, incompatibility_coefficient, scope='global'):
        base.Weight.__init__(self, locus1=locus1, locus2=locus2, incompatibility_coefficient=incompatibility_coefficient, _scope=scope)
        labels = { 'axes': ["offspring background-%s" % locus1, "offspring background-%s" % locus2], \
                   'elements': [["o%s1" % locus1, "o%s2" % locus1], \
                                ["o%s1" % locus2, "o%s2" % locus2]] }
        self._init_labels(labels)
    
    def _build(self):
        h = self.incompatibility_coefficient
        ary = np.array( [[1, 1-h], \
                        [1-h, 1]], float )
        return ary

class MatingReadyness1Locus(base.Weight):
    """Readyness for mating (e.g. timing) is determined by one nuclear locus.
       Overlap between different distributions gives probability of mating.
       Parameter 'separation' is one minus overlap.
    """
    _type = 'constant reproduction'
    _info = 'mating readyness'
    _parameter_names = ['separation']
    labels = { 'axes': ['female readyness', 'male readyness'], \
               'elements': [['fS1','fS2'],['mS1','mS2']] }
    
    def __init__(self, separation, scope='global'):
        base.Weight.__init__(self, separation=separation, _scope=scope)
    
    def _build(self):
        ary = np.array( [[1, 1-self.separation], [1-self.separation, 1]], float )
        return ary

def main():
    # weights:
    CI = CytoplasmicIncompatibility(0.9)
    F = FecundityReduction(0.)
    NIA = NuclearInheritanceAtSingleLocus('neutral', 'N', range(1,4))
    NITP = NuclearInheritanceAtTwoLoci(0.5, 'trait', 'T', range(1,3), 'preference', 'P', range(2))
    MP = MatingPreference(0.99, 'pr', 'P0','None',0., 'P1','T1',0.99)
    VS = ViabilitySelection(selection_coefficients=[1.,0.])
    VSM = ViabilitySelectionMales(selection_coefficients=[1.,0.])
    T = WolbachiaTransmission(0.87)
    
    # sexually implicit:
    freqs = numpytools.random_frequencies((2,2,3,2))
    labels = dict(axes=['trait','preference','neutral','cytotype'], \
                  elements=[['T1','T2'],['P0','P1'],['N1','N2','N3'],['U','W']])
    pop = base.Population(freqs, labels)
    
    pop.add_weights(CI, F, NIA, NITP, MP, VS, T)
    
    return pop
    
    ## sexually explicit:
    #freqs_si = numpytools.random_frequencies((2,3,3,3,2))
    #labels_si = dict(axes=['sex','trait','preference','neutral','cytotype'], \
                  #elements=[['female','male'],['T1','T2','T3'],['P0','P1','P2'],['A1','A2','A3'],['uninfected','Wolbachia']])
    #pop_si = base.Population(freqs_si, labels_si)
    
    #pop_si.add_weights(CI, F, NIA, NITP, MP, VS, T)
    
    #print pop_si
    #print pop_si.weights


if __name__ == '__main__':
    p = main()
