import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import utilities as utils
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import colorbrewer as cb
from mpltools import color

legend_font = FontProperties()
legend_font.set_size('small')
ticks_font = FontProperties(family='Helvetica', style='normal', \
                            size=8, weight='normal', stretch='normal')
empty_rect = Rectangle((0,0),1,1)
empty_rect.set_alpha(0)
empty_rect.set_edgecolor('white')
color_scheme_names = ['BrBG', 'PiYG', 'PRGn', 'RdBu', 'RdGy', 'PuOr', \
    'RdYlBu', 'RdYlGn', 'Spectral']
color_schemes = {}
for name in color_scheme_names:
    color_schemes[name] = eval('cb.{0}'.format(name))

def to_rgb(t):
    """
    Args:
        t: 3-tuple of int's in range [0,255]
    
    Returns:
        out: 3-tuple of float's in range [0,1]
    """
    r,g,b = np.array(t)/255.
    return (r, g, b)

def plot_sums(gens, sums, c, loci, alleles, figsize=[19,8], **kwargs):
    pops = alleles[0]
    alleles = alleles[1:]
    if loci[0] == 'population':
        loci = loci[1:]
    npops = len(pops)   # -> subplot columns
    nloci = len(loci)   # -> subplot rows
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.93, hspace=0.2, wspace=0.05)
    fig.text(0.02, 0.49, 'frequency', fontsize=12, ha='center', va='center', rotation='vertical')
    fig.text(0.47, 0.04, 'generation', fontsize=12, ha='center', va='center', rotation='horizontal')
    xmax = gens[c-1]
    loclabel_xpos = xmax/100.
    for i,pop in enumerate(pops):
        for j,loc in enumerate(loci):
            sno = i+1+j*npops    # subplot number
            ax = fig.add_subplot(nloci, npops, sno)
            ax.grid(True)
            ax.text(loclabel_xpos, 1.05, loc, fontsize=8)
            if j == 0:
                ax.set_title(pop)
            # default: neither x nor y ticklabels:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            n = len(alleles[j])     # number of alleles at the locus
            name = color_scheme_names[j]
            loc_scheme = color_schemes[name][max(4,n)]     # color schemes should have at least 4 colors
            if i == 0:      # y ticklabels only for pop1:
                plt.setp(ax.get_yticklabels(), visible=True)
                plt.setp(ax.get_yticklabels(), fontsize=8)
            if j == nloci-1:    # x ticklabels only for last locus:
                plt.setp(ax.get_xticklabels(), visible=True)
                plt.setp(ax.get_xticklabels(), fontsize=8)
            for k,allele in enumerate(alleles[j]):   # iterate through alleles at the locus/lines
                allele_color = to_rgb(loc_scheme[k])
                ax.plot(gens[:c], sums[loc][:c,i,k], color=allele_color, label=allele, **kwargs)
                ax.set_xlim(0,xmax)
                ax.set_ylim(-0.03,1.03)           # all data to plot are frequencies, i.e. between 0 and 1
            if i == npops-1:    # one legend is enough
                leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))   # legend outside of axes at the right
                leg.get_frame().set_alpha(0) # this will make the box totally transparent
                leg.get_frame().set_edgecolor('white')
    return fig

def stacked_bars(sums, loci, alleles, generation=None, figsize=[15,8]):
    """
    Args:
        sums: list of ndarrays
            alleles sums for each locus
        loci: list of strings
            loci names (including `pop`)
        alleles: list of list of strings
            allele names (including `populations`)
    """
    # some calculations we need:
    pops = alleles[0]
    npops = len(pops)
    alleles = alleles[1:]
    ashape = utils.list_shape(alleles)
    ntotal = np.sum(ashape)   # total number of alleles
    maxalleles = max(ashape)  # maximum number of alleles at one locus
    loci = loci[1:]     # we don't need the population locus name
    nloci = len(loci)
    width = 1.     # bar width
    xpos = np.arange(nloci)
    
    # prepare data:
    data = np.zeros((npops,nloci,maxalleles), float)
    for i,loc in enumerate(sums):
        p,l = loc.shape
        data[:,i,:l] = loc        # pop, locus, allele
    cumdata = np.cumsum(data, axis=2)   # we need this for stacking the bars
    
    # prepare figure and axes and plot:
    fig = plt.figure(figsize=figsize)
    if generation is not None:
        fig.text(0.94, 0.02, 'Generation: {0}'.format(generation), fontsize=12, ha='right', va='bottom', rotation='horizontal')
    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.94, top=0.91, bottom=0.12)
    #~ fig.subplots_adjust(bottom=0.3)
    for i,pop in enumerate(pops):
        ax = fig.add_subplot(1,npops,i+1)
        ax.grid(False)
        ax.set_title(pop)
        if i==0:
            ax.set_ylabel('frequency')
        ax.set_xticks(xpos+width/2.)
        ax.set_xticklabels(loci)
        for j,loc in enumerate(loci):
            n = len(alleles[j])     # number of alleles at the locus
            name = color_scheme_names[j]
            loc_scheme = color_schemes[name][max(4,n)]
            for k,allele in enumerate(alleles[j]):
                allele_color = to_rgb(loc_scheme[k])
                if k==0:
                    ax.bar(xpos[j], data[i,j,k], width, color=allele_color, label=allele)
                else:
                    ax.bar(xpos[j], data[i,j,k], width, color=allele_color, bottom=cumdata[i,j,k-1], label=allele)
        ax.set_ylim(0,1)
        if i==npops-1:
            handles, labels = ax.get_legend_handles_labels()
            cumshape = np.cumsum(ashape)[::-1]
            for idx in cumshape:
                labels.insert(idx, ' ')
                handles.insert(idx, empty_rect)
            leg = plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.01), prop=legend_font)
            leg.get_frame().set_alpha(0) # this will make the box totally transparent
            leg.get_frame().set_edgecolor('white')
    fig.autofmt_xdate()    # automatic label rotation
    fig.subplots_adjust(bottom=0.12)  # we need to re-apply bottom spacing after label rotation!
    return fig
