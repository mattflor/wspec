"""
.. module:: analytical
   :platform: Unix
   :synopsis: Analytical solutions

.. moduleauthor:: Matthias Flor <matthias.c.flor@gmail.com>

"""

from numpy import power,sqrt

# Stability of CI

### Convenience functions ##############################################
def _R(f,ci,t):
    """
    Convenience function. Usually, there should be no need to
    call this function directly.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
    """
    return power(ci-f,2) - 4 * power(1-f,2) * ci * t * (1-t)

def _Rh(f,ci,m,t):
    """
    Convenience function.  Usually, there should be no need to
    call this function directly.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
    """
    return power(ci-f,2) - 4 * (1-f) * ci * t * ( m + (1-f)*(1-m)*(1-t) )

def _D(f,ci,t):
    """
    Convenience function. Usually, there should be no need to
    call this function directly.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
    """
    return 2 * ci * ( f + (1-f)*t )

def _P(f,ci,m,t):
    """
    Convenience function. Usually, there should be no need to
    call this function directly.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
    """
    return (f+ci) * (1+m) - (1-m) * sqrt(_R(f,ci,t))

def _Q(f,t):
    """
    Convenience function. Usually, there should be no need to
    call this function directly.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
    """
    return 2 * power(1-f,2) * t * (1-t)

def _Qh(f,m,t):
    """
    Convenience function. Usually, there should be no need to
    call this function directly.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
    """
    return 2 * (1-f) * ( (1 - f*(1-m)) * t - (1-f)*(1-m)*power(t,2) )

### Single host population #############################################
# Dynamics:
def dynamics_SP(f,ci,t,x):
    """
    Infection dynamics of Wolbachia in a single host population (SP), 
    iterative map returns the infection frequency in the next
    generation.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        x: float in interval [0, 1]
            infection frequency of Wolbachia
        
    Returns:
        out: float
            infection frequency in the next generation
    """
    return (1-f)*t*x / (1-f*x*(1-ci*(1-t)*x)-ci*x*(1-t*x))
    
# Fixpoints:
def fix1_SP(f,ci,t):
    """
    Infection frequency fixpoint :math:`x_1^{\\ast}` for a single host population (SP).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
            equilibrium frequency of Wolbachia
    """
    return 0.

def fix2_SP(f,ci,t):
    """
    Infection frequency fixpoint :math:`x_2^{\\ast}` for a single host population (SP).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
            equilibrium frequency of Wolbachia
    """
    numer = f + ci - sqrt(_R(f,ci,t))
    denom = _D(f,ci,t)
    return numer/denom

def fix3_SP(f,ci,t):
    """
    Infection frequency fixpoint :math:`x_3^{\\ast}` for a single host population (SP).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
            equilibrium frequency of Wolbachia
    """
    numer = f + ci + sqrt(_R(f,ci,t))
    denom = _D(f,ci,t)
    return numer/denom

def fix_SP(f,ci,t):
    """
    Infection frequency fixpoints for a single host population (SP).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: tuple of three floats
            equilibrium frequencies :math:`x_1^{\\ast}`, :math:`x_2^{\\ast}`, and :math:`x_3^{\\ast}` of Wolbachia
    """
    root = sqrt(_R(f,ci,t))
    numer2 = f + ci - root
    numer3 = f + ci + root
    denom = _D(f,ci,t)
    return 0., numer1/denom, numer3/denom

# Critical CI level:
def lcrit_SP(f,t):
    """
    Critical CI level (lcrit) for a single host population (SP).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: float
            critical CI level
    """
    q = _Q(f,t)
    return f + q + sqrt( power(f+q,2) - power(f,2) )

### Uninfected mainland ################################################
# Dynamics:
def dynamics_UM(f,ci,m,t,x):
    """
    Infection dynamics of Wolbachia for the scenario with an 
    uninfected mainland (UM), iterative map returns the infection 
    frequency in the next generation.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        x: float in interval [0, 1]
            infection frequency of Wolbachia
    
    Returns:
        out: float
            infection frequency in the next generation
    """
    return (1-m)*dynamics(f,ci,t,x)
    
# Fixpoints:
def fix1_UM(f,ci,m,t):
    """
    Fixpoint :math:`x_1^{\\ast}` for the scenario with an uninfected mainland (UM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            equilibrium frequency of Wolbachia
    """
    return 0.

def fix2_UM(f,ci,m,t):
    """
    Fixpoint :math:`x_2^{\\ast}` for the scenario with an uninfected mainland (UM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            equilibrium frequency of Wolbachia
    """
    numer = f + ci - sqrt(_Rh(f,ci,m,t))
    denom = _D(f,ci,t)
    return numer/denom

def fix3_UM(f,ci,m,t):
    """
    Fixpoint :math:`x_3^{\\ast}` for the scenario with an uninfected mainland (UM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            equilibrium frequency of Wolbachia
    """
    numer = f + ci + sqrt(_Rh(f,ci,m,t))
    denom = _D(f,ci,t)
    return numer/denom

def fix_UM(f,ci,m,t):
    """
    Infection frequency fixpoints for the scenario with an uninfected 
    mainland (UM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: tuple of three floats
            equilibrium frequencies :math:`x_1^{\\ast}`, :math:`x_2^{\\ast}`, and :math:`x_3^{\\ast}` of Wolbachia
    """
    root = sqrt(_Rh(f,ci,t))
    numer2 = f + ci - root
    numer3 = f + ci + root
    denom = _D(f,ci,t)
    return 0., numer2/denom, numer3/denom

# Critical CI level:
def lcrit_UM(f,m,t):
    """
    Critical CI level (lcrit) for the scenario with an uninfected 
    mainland (UM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical CI level
    """
    q = _Qh(f,m,t)
    return f + q + sqrt( power(f+q,2) - power(f,2) )

# Critical migration rate:
def mcrit_UM(f,ci,t):
    """
    Critical migration rate (mcrit) for the scenario with an uninfected 
    mainland (UM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical migration rate
    """
    numer = _R(f,ci,t)
    denom = 2 * (1-f) * t * _D(f,ci,t)
    return numer/denom

### Infected mainland ##################################################
# Dynamics:
def dynamics_IM(f,ci,m,t,x):
    """
    Infection dynamics of Wolbachia for the scenario with an infected 
    mainland (IM), iterative map.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        x: float in interval [0, 1]
            infection frequency of Wolbachia
    
    Returns:
        out: float
            infection frequency in the next generation
    """
    return (1-m)*dynamics(f,ci,t,x) + m*fix3_SP(f,ci,t)
    
# Fixpoints:
def fix1_IM(f,ci,m,t):
    """
    Fixpoint :math:`x_1^{\\ast}` for the scenario with an infected mainland (IM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical migration rate
    """
    return fix3_SP(f,ci,t)

def fix2_IM(f,ci,m,t):
    """
    Fixpoint :math:`x_2^{\\ast}` for the scenario with an infected mainland (IM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical migration rate
    """
    p = _P(f,ci,m,t)
    d = _D(f,ci,t)
    numer = p + sqrt( power(p - 8 * m * d ) )
    denom = 2*d
    return numer/denom

def fix3_IM(f,ci,m,t):
    """
    Fixpoint :math:`x_3^{\\ast}` for the scenario with an infected mainland (IM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
            
    Returns:
        out: float
            critical migration rate
    """
    p = _P(f,ci,m,t)
    d = _D(f,ci,t)
    numer = p - sqrt( power(p - 8 * m * d ) )
    denom = 2*d
    return numer/denom
    
def fix_UM(f,ci,m,t):
    """
    Infection frequency fixpoints for the scenario with an infected 
    mainland (IM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        m: float in interval [0, 1]
            migration rate
        t: float in interval [0, 1]
            transmission rate of Wolbachia
    
    Returns:
        out: tuple of three floats
            equilibrium frequencies :math:`x_1^{\\ast}`, :math:`x_2^{\\ast}`, and :math:`x_3^{\\ast}` of Wolbachia
    """
    p = _P(f,ci,m,t)
    d = _D(f,ci,t)
    root = sqrt( power(p,2) - 8 * m * d )
    numer2 = p + root
    numer3 = p - root
    denom = 2*d
    return fix3_SP(f,ci,t), numer2/denom, numer3/denom

# Critical migration rate:
def mcrit_IM(f,ci,t):
    """
    Critical migration rate (mcrit) for the scenario with an infected 
    mainland (IM).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical migration rate
    """
    numer = 1 - sqrt((1-f)*t)
    denom = f + ci + sqrt(_R(f,ci,t))
    return 2 * _D(f,ci,t) * power( numer/denom, 2 )

### Local host adaptation ######################################################
# Critical migration rates:
def mcrit_UMA(f,ci,s,t):
    """
    Critical migration rate (mcrit) for the scenario with an uninfected 
    mainland (UM) with local host adaptation (A).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        s: positive float
            selection coefficient
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical migration rate
    """
    numer = (1+s) * _R(f,ci,t)
    denom = 2 * (1-f) * t * _D(f,ci,t)  +  s * _R(f,ci,t)
    return numer/denom

def mcrit_IMA(f,ci,s,t):
    """
    Critical migration rate (mcrit) for the scenario with an infected 
    mainland (IM) with local host adaptation (A).
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        s: positive float
            selection coefficient
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            critical migration rate
    """
    d = _D(f,ci,t)
    root = sqrt(_R(f,ci,t))
    numer = (1+s) * ( 2 * d * power(1-sqrt((1-f)*t),2) + s * power(f+ci-root,2) )
    denom = power( (f+ci)*(1-s) + (1+s)*root, 2 ) + 8 * s * d
    return  numer/denom


################################## Gene flow ###################################
# Divergent selection:
def reproval_DS(s):
    """
    Reproductive value of a migrant for the case of divergent 
    selection (DS). Residents have a viability advantage of `s` over 
    migrants, equivalent to migrants having a viability cost of 
    :math:`\\frac{s}{1+s}`. Derived from fitness graph.
    
    Args:
        s: positive float
            selection coefficient
        
    Returns:
        out: float
            reproductive value
    """
    return 1./(1+2*s)

def meff_DS(m,s):
    """
    Effective migration rate (meff) for the case of divergent selection 
    (DS). Residents have a viability advantage of `s` over migrants, 
    equivalent to migrants having a viability cost of :math:`\\frac{s}{1+s}`.
    Derived from fitness graph for the reproductive value of a migrant.
    
    Args:
        m: float in interval [0, 1]
            migration rate
        s: positive float
            selection coefficient
        
    Returns:
        out: float
            effective migration rate
    """
    return m * reproval_DS(s)

def gff_DS(m,s):
    """
    Gene flow factor (gff) for the case of divergent selection (DS).
    Residents have a viability advantage of `s` over migrants, 
    equivalent to migrants having a viability cost of :math:`\\frac{s}{1+s}`.
    Derived from fitness graph for the reproductive value of a migrant.
    
    Args:
        m: float in interval [0, 1]
            migration rate
        s: positive float
            selection coefficient
        
    Returns:
        out: float
            gene flow factor
    """
    return meff_DS(s) / m
    
# unidirectional CI:
# homogenous infected population:    
def reproval_FUHOMW(f,ci):
    """
    Reproductive value of an uninfected female migrant (FU) in a 
    homogenous Wolbachia infected population (HOMW). This applies to 
    the case of perfect Wolbachia transmission, :math:`t=1`. Derived from 
    fitness graph:
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        
    Returns:
        out: float
            reproductive value
    """
    return (1-ci) / (1 - 2*f + ci)

def reproval_MUHOMW(f,ci):
    """
    Reproductive value of an uninfected male migrant (MU) in a 
    homogenous Wolbachia infected population (HOMW). This applies to 
    the case of perfect Wolbachia transmission, t=1.
    Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        
    Returns:
        out: float
            reproductive value
    """
    return 1

def reproval_UHOMW(f,ci):
    """
    Reproductive value of an uninfected migrant (U) in a a 
    homogenous Wolbachia infected population (HOMW). This applies to 
    the case of perfect Wolbachia transmission, t=1. Derived from 
    fitness graph. Averaged over the two sexes.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        
    Returns:
        out: float
            reproductive value
    """
    return (1-f) / (1 - 2*f + ci)

# homogenous uninfected population:
def reproval_FWHOMU(f,ci,t):
    """
    Reproductive value of a Wolbachia infected female migrant (FW)
    in a homogenous uninfected population (HOMU). Derived from fitness 
    graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return (1-f)*(2-(1+ci)*t) / (2-(1-f)*t)

def reproval_MWHOMU(f,ci,t):
    """
    Reproductive value of a Wolbachia infected male migrant (MW) 
    in a homogenous uninfected population (HOMU). Derived from fitness 
    graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return 1-ci

def reproval_WHOMU(f,ci,t):
    """
    Reproductive value of a Wolbachia infected migrant (W) in a 
    homogenous uninfected population (HOU). Derived from fitness graph.
    Averaged over the two sexes.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return 1 - (f+ci) / (2-(1-f)*t)

def reproval_IPHOMU(f,ci,t):
    """
    Average reproductive value of a migrant from an infected 
    population (IP), i.e. where the Wolbachia infection is at a stable 
    frequency below 1 due to imperfect transmission, in a homogenous 
    uninfected population (HOU). Averaged over sexes (F/M) and 
    cytotypes (U/W). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    x =fix3_SP(f,ci,t)
    return 1 - x*(f+ci) / (2-(1-f)*t)

# heterogenous population:
def reproval_FUHETCP(f,ci,t):
    """
    Reproductive value of an uninfected female (FU) in a 
    heterogenous host population (HET), `per capita` (CP) reproductive 
    value (as opposed to `per class`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    root = sqrt(_R(f,ci,t))
    return (2*(-1+ci)*ci*(-1+f)*t)/(f*(f+root)+power(ci,2)*(1+(-1+f)*t)-ci*(root+power(f,2)*(5-6*t)*t+(4-3*root-6*t)*t+f*(2+3*t*(-3+root+4*t))))

def reproval_FUHETCL(f,ci,t):
    """
    Reproductive value of an uninfected female (FU) in a 
    heterogenous host population (HET), `per class` (CL) reproductive 
    value (as opposed to `per capita`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return reproval_FUHETCP(f,ci,t) * (1-fix_3SP(f,ci,t)) / 2.

def reproval_MUHETCP(f,ci,t):
    """
    Reproductive value of an uninfected male (MU) in a heterogenous 
    host population (HET), `per capita` (CP) reproductive value (as 
    opposed to `per class`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    root = sqrt(_R(f,ci,t))
    return (ci*(power(ci,2)+(-1+f)*(f+root)*(1+2*f*(-1+t)-2*t)*t+ci*(-f+root-3*t+(7-4*f)*f*t+4*power((-1+f)*t,2))))/(power(ci,3)+power(f,2)*(f+root)+power(ci,2)*(-f+root-4*t+(9-5*f)*f*t+5*power((-1+f)*t,2))+ci*(5*power(f,3)*(-1+t)*t+root*t*(-2+3*t)+f*t*(-4+5*root+5*t-6*root*t)+power(f,2)*(-1+t*(9-3*root-10*t+3*root*t))))

def reproval_MUHETCL(f,ci,t):
    """
    Reproductive value of an uninfected male (MU) in a heterogenous 
    host population (HET), `per class` (CL) reproductive value (as 
    opposed to `per capita`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return reproval_MUHETCP(f,ci,t) * (1-fix3SP(f,ci,t)) / 2.

def reproval_FWHETCP(f,ci,t):
    """
    Reproductive value of a Wolbachia infected female (FW) in a 
    heterogenous host population (HET), `per capita` (CP) reproductive 
    value (as opposed to `per class`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    root = sqrt(_R(f,ci,t))
    return -(4*ci*(power(ci,4)*(1+2*f*(-1+t)-2*t)+(-1+f)*f*(f+root)*(1+2*f*(-1+t)-2*t)*(power(f,2)*(-1+t)-t)*t+power(ci,3)*(root-2*(3+root)*t+13*power(f,3)*power(-1+t,2)*t+17*power(t,2)-13*power(t,3)+power(f,2)*(4+t*(-34+69*t-39*power(t,2)))+f*(-2*(1+root)+t*(27+2*root-60*t+39*power(t,2))))+ci*(-1+f)*t*(10*power(f,5)*power(-1+t,3)*t+power(f,4)*power(-1+t,2)*(-1+3*t*(9-2*root+2*(-6+root)*t))+t*(-3*root+t*(2+9*root+t*(-6-11*root+4*t+6*root*t)))-f*t*(2-11*root+t*(1+39*root+3*t*(-3-17*root+2*t+8*root*t)))-power(f,3)*(-1+t)*(root+t*(4*(-7+4*root)+t*(67-41*root-44*t+24*root*t)))+power(f,2)*(-root+t*(11-17*root+t*(-35+69*root+t*(39-87*root+4*(-4+9*root)*t)))))+power(ci,2)*(20*power(f,5)*power(-1+t,3)*power(t,2)-power(f,4)*power(-1+t,2)*t*(8+t*(-81+100*t))+t*(-4*root+t*(9+11*root-t*(31+9*root+t*(-41+20*t))))+power(f,3)*(-1+t)*(2+t*(-9*(2+root)+t*(157+9*root+4*t*(-81+50*t))))+power(f,2)*(1+2*root-t*(16+22*root+t*(-148-47*root+t*(423+27*root-486*t+200*power(t,2)))))+f*(-root+t*(4+17*root+t*(-59-40*root+t*(3*(61+9*root)+4*t*(-56+25*t))))))))/((ci+f+root)*(ci-f+root-2*t+2*(3-2*f)*f*t+4*power((-1+f)*t,2))*(power(ci,3)+power(f,2)*(f+root)+power(ci,2)*(-f+root-4*t+(9-5*f)*f*t+5*power((-1+f)*t,2))+ci*(5*power(f,3)*(-1+t)*t+root*t*(-2+3*t)+f*t*(-4+5*root+5*t-6*root*t)+power(f,2)*(-1+t*(9-3*root-10*t+3*root*t)))))

def reproval_FWHETCL(f,ci,t):
    """
    Reproductive value of an infected female (FW) in a heterogenous 
    host population (HET), `per class` (CL) reproductive value (as 
    opposed to `per capita`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return reproval_FWHETCP(f,ci,t) *fix3_SP(f,ci,t) / 2.

def reproval_MWHETCP(f,ci,t):
    """
    Reproductive value of a Wolbachia infected male (MW) in a 
    heterogenous host population (HET), `per capita` (CP) reproductive 
    value (as opposed to `per class`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    root = sqrt(_R(f,ci,t))
    return (4*ci*(power(ci,4)+(-1+f)*f*(f+root)*(1+2*f*(-1+t)-2*t)*t*(power(f,2)+(-1+(4-3*f)*f)*t+2*power((-1+f)*t,2))+power(ci,3)*(-2*f+root-6*t-f*(-13+f*(6+f))*t+power(-1+f,2)*(7+2*f)*power(t,2)-power(-1+f,3)*power(t,3))+power(ci,2)*(-4*power(f,5)*power(-1+t,3)*power(t,2)+power(f,4)*power(-1+t,2)*power(t,2)*(3+20*t)-power(f,3)*(-1+t)*t*(6-root+t*(-41+root+4*t*(3+10*t)))+t*(-4*root+t*(9+5*root+t*(-23+root+t*(11+4*t))))-f*(root+t*(-4-9*root+t*(47+8*root+t*(-91+3*root+4*t*(8+5*t)))))+power(f,2)*(1+t*(-2*(5+2*root)+t*(78+root+t*(3*(-41+root)+2*t*(9+20*t))))))+ci*(-1+f)*t*(2*power(f,5)*power(-1+t,3)*t*(-3+4*t)-t*(-1+2*t)*(-3*root+t*(2+3*root+t*(-6+root+4*t)))-power(f,4)*power(-1+t,2)*(-1+t*(11-2*root+2*t*(-28+root+20*t)))+power(f,3)*(-1+t)*(root+t*(-4*(1+root)+t*(83-5*root+4*t*(-41+2*root+20*t))))+f*t*(-2+13*root+t*(-11-29*root+t*(67+9*root+2*t*(-47+4*root+20*t))))+power(f,2)*(root+t*(5-17*root+t*(43+25*root+t*(3*(-61+root)-4*t*(-54+3*root+20*t))))))))/((ci+f+root)*(ci-f+root-2*t+2*(3-2*f)*f*t+4*power((-1+f)*t,2))*(power(ci,3)+power(f,2)*(f+root)+power(ci,2)*(-f+root-4*t+(9-5*f)*f*t+5*power((-1+f)*t,2))+ci*(5*power(f,3)*(-1+t)*t+root*t*(-2+3*t)+f*t*(-4+5*root+5*t-6*root*t)+power(f,2)*(-1+t*(9-3*root-10*t+3*root*t)))))

def reproval_MWHETCL(f,ci,t):
    """
    Reproductive value of an infected male (MW) in a heterogenous 
    host population (HET), `per class` (CL) reproductive value (as 
    opposed to `per capita`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return reproval_MWHETCP(f,ci,t) *fix3_SP(f,ci,t) / 2.

def reproval_UHETCP(f,ci,t):
    """
    Average reproductive value of an uninfected host (U) in a 
    heterogenous population (HET), `per capita` (CP) reproductive 
    value (as opposed to `per class`). Derived from fitness graph. 
    Averaged over the two sexes.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    root = sqrt(_R(f,ci,t))
    return ((-1+f)*t*(power(ci,2)+f*(f-root)-ci*(root+4*t+2*(f+f*(-5+3*f)*t-3*power((-1+f)*t,2)))))/(2*(power(ci-f,2)+2*(-1+f)*(ci*(2+ci)-4*ci*f+power(f,2))*t-3*ci*power(-1+f,2)*(-4+3*f)*power(t,2)+9*ci*power((-1+f)*t,3)))

def reproval_UHETCL(f,ci,t):
    """
    Average reproductive value of an uninfected host (U) in a 
    heterogenous population (HET), `per class` (CL) reproductive value 
    (as opposed to `per capita`). Derived from fitness graph. 
    Averaged over the two sexes.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return reproval_UHETCP(f,ci,t) * (1-fix3SP(f,ci,t))

def reproval_WHETCP(f,ci,t):
    """
    Average reproductive value of a Wolbachia infected host (W) in 
    a heterogenous population (HET), `per capita` (CP) reproductive 
    value (as opposed to `per class`). Derived from fitness graph. 
    Averaged over the two sexes.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    root = sqrt(_R(f,ci,t))
    return (4*ci*(f*(-1+t)-t)*(-power(ci,4)+(-1+f)*f*(f+root)*(1+2*f*(-1+t)-2*t)*(f*(-1+t)-t)*t-power(ci,3)*(-2*f+root-5*t+(12-7*f)*f*t+7*power((-1+f)*t,2))-ci*(-1+f)*t*(-4*power(f,4)*(-2+t)*power(-1+t,2)*t+power(t,2)*(-2+3*root-2*t*(-3+2*root+2*t))+power(f,3)*(-1+t)*(-1+t*(19-4*root+2*t*(-19+2*root+8*t)))+f*(root+(-1+t)*t*(3-2*root+2*t*(-9+6*root+8*t)))-power(f,2)*(root+t*(-15+6*root+t*(56-19*root+6*t*(-11+2*root+4*t)))))-power(ci,2)*(12*power(f,4)*power((-1+t)*t,2)+power(f,3)*t*(4+t*(-43+87*t-48*power(t,2)))+t*(-3*root+t*(4+5*root+3*t*(-5+4*t)))-f*(root+t*(-2-8*root+t*(27+10*root-69*t+48*power(t,2))))+power(f,2)*(1+t*(-6-5*root+t*(54+5*root+9*t*(-13+8*t)))))))/((ci+f+root)*(ci-f+root-2*t+2*(3-2*f)*f*t+4*power((-1+f)*t,2))*(power(ci,3)+power(f,2)*(f+root)+power(ci,2)*(-f+root-4*t+(9-5*f)*f*t+5*power((-1+f)*t,2))+ci*(5*power(f,3)*(-1+t)*t+root*t*(-2+3*t)+f*t*(-4+5*root+5*t-6*root*t)+power(f,2)*(-1+t*(9-3*root-10*t+3*root*t)))))

def reproval_WHETCL(f,ci,t):
    """
    Reproductive value of an infected host (W) in a heterogenous host 
    population (HET), `per class` (CL) reproductive value (as opposed to `per 
    capita`). Derived from fitness graph.
    
    Args:
        f: float in interval [0, 1]
            fecundity reduction in infected females
        ci: float in interval [0, 1]
            level of CI
        t: float in interval [0, 1]
            transmission rate of Wolbachia
        
    Returns:
        out: float
            reproductive value
    """
    return reproval_WHETCP(f,ci,t) *fix3_SP(f,ci,t)

################################## Reinforcement ##############################################
def xPref(ci,pr,q,s):
    """
    Approximated equilibrium frequency of a mutant allele at the locus 
    for female mating preference in an uninfected island receiving 
    migration from an infected mainland.
    
    Args:
        ci: float in interval [0, 1]
            level of CI
        pr: float in interval [0, 1]
            rejection probability of mating preference mutant allele
        q: float in interval [0, 1]
            transition probability to a new mating round
        s: positive float
            selection coefficient
         
    Returns:
        out: float
            equilibrium frequency of mutant preference allele
    """
    return 1 - (1-ci)*s/((2*ci*s - (1-ci)*(1-q))*pr)

def prcrit(ci,q,s):
    """
    Minimal rejection probability (pr) of a mutant allele at the locus 
    for female mating preference to spread in an uninfected island 
    receiving migration from an infected mainland.
    
    Args:
        ci: float in interval [0, 1]
            level of CI
        q: float in interval [0, 1]
            transition probability to a new mating round
        s: positive float
            selection coefficient
         
    Returns:
        out: float
            minimal rejection probability
    """
    return (1-ci)*s / (2*ci*s - (1-ci)*(1-q))

def xTprefEQ(pr,s,xPref):
    """
    Modified from Kirkpatrick (1982), equation (2).
    
    Args:
        pr: float in interval [0, 1]
            rejection probability of mating preference mutant allele
        s: positive float
            selection coefficient
        xPref: float in interval [0, 1]
            frequency of preference allele in Kirkpatrick's model
    
    Returns:
        out: float
            equilibrium frequency of preferred trait
    """
    # calculate Kirkpatrick's preference strength (ak) from rejection 
    # probability (pr):
    ak = 1./(1-pr)    
    # calculate Kirkpatrick's viability cost (sk) from selection 
    # coefficient (s):
    sk = s/(1.+s)
    # Kirkpatrick used a male-limited trait, but in our model, the trait is 
    # expressed in both sexes:
    p2 = xPref/2.

    if p2 <= sk / ((ak-1)*(1-sk)):
        return 0.
    elif p2 >= ak*sk / (ak-1):
        return 1.
    else:
        return (1./sk + 1./(ak*(1-sk)-1))*p2 - 1./(ak*(1-sk)-1)
    
    
