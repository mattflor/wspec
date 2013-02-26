import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


legend_font = FontProperties()
legend_font.set_size('small') 

def create_figs(pops, loci, figsize=[5,7]):
    """
    Args:
        pops: list of strings
            list of population names; used for figure suptitles
        loci: list of strings
            list of locus names (excluding `pops`!); used for axes titles
    """
    npops = len(pops)
    nloci = len(loci)
    figs = []
    for i,pop in enumerate(pops):
        fig = plt.figure(figsize=figsize)
        figs.append(fig)
        fig.subplots_adjust(left=0.11, right=0.82, bottom=0.07, top=0.94, hspace=0.14)
        fig.suptitle(pop)
        fig.text(0.04, 0.5, 'frequency', ha='center', va='center', rotation='vertical')
        ax1 = fig.add_subplot(nloci,1,nloci)    # make the bottom axes the one to share with
        plt.setp(ax1.get_xticklabels(), fontsize=8)
        plt.setp(ax1.get_yticklabels(), fontsize=8)
        ax1.grid()
        ax1.set_xlabel('generation')
        ax1.text(0, 1.01, loci[-1], fontsize=8)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01), prop=legend_font)
        for j in range(nloci-1,0,-1):
            ax = fig.add_subplot(nloci,1,j)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            ax.grid()
            ax.text(0, 1.02, loci[j-1], fontsize=8)
        plt.draw()
    return figs

def plot_sums(gens, sums, c, loci, alleles, figs, **kwargs):
    """
    Args:
        gens: ndarray of ints
            generations
        sums: HDF5 group
            the group must contain loci sums arrays
        c: int
            generation counter
        loci: list of strings
            list of locus names (excluding `pops`!)
        alleles: nested list of strings
            allele names used for the legends
        figs: list of matplotlib figures
            created by function `create_figs`
    """
    npops = len(alleles[0])
    for i in range(npops):
        nloci = len(loci)
        for j,loc in enumerate(loci):
            for k,allele in enumerate(alleles[j+1]):
                ax = figs[i].get_axes()[nloci-1-j]
                ax.plot(gens[:c], sums[loc][:c,i,k], label=allele, **kwargs)
                ax.set_xlim(0,gens[c-1])
                ax.set_ylim(0,1)
                ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01), prop=legend_font)
    plt.show()

#~ class stripchart(object):
    #~ def __init__(self, labels):
        #~ """
        #~ Args:
            #~ labels: nested list of strings
                #~ equivalent to `ALLELES` in a scenario file
        #~ """
        #~ # set up dimensions:
        #~ self.pops = labels[0]
        #~ self.loci = labels[1:]
        #~ self.shape = utils.list_shape(labels)
        #~ n_pops = len(self.pops)       # --> number of figures
        #~ n_loci = len(self.loci)       # --> number of subplots per figure
        #~ ny,nx = loc_dims[str(n_loci)]
        #~ figsize = [n_cols,n_rows+1]
        #~ 
        #~ self.figs = []
        #~ self.data = []
        #~ for pop in self.pops:
            #~ self.figs.append(figure(figsize=figsize))
            #~ self.data
            #~ for loc in self.loci:
                #~ 
            #~ 
        #~ 
        #~ self.fig = plt.figure(figsize=figsize)
        #~ if title:
            #~ self.fig.suptitle(title)
        #~ self.fig.subplots_adjust(left = 0.04, right = 0.94, wspace = 0.32)
        #~ legend_font = FontProperties()
        #~ legend_font.set_size('small')       
        #~ self.canvas = self.fig.canvas
        #~ self.figaxes = []
        #~ self.axlines = []
        #~ self.labels = labels
        #~ self.populations = sorted(labels.keys())
        #~ self.n_pops = len(self.populations)
        #~ self.xmin, self.xmax = 0, 100    # same for all axes
        #~ 
        #~ for i,pop in enumerate(self.populations):      # population counter, name
            #~ ax = self.fig.add_subplot(1, self.n_pops, i+1)
            #~ self.figaxes.append(ax)
            #~ self.axlines.append([])
            #~ for j,al in enumerate(self.labels[pop]):   # allele counter, label
                #~ line, = ax.plot([], [], lw=2, label=al)  # animated=True, 
                #~ self.axlines[i].append(line)
                #~ #                print ax.get_xlim()
            #~ ax.set_title(pop)
            #~ ax.set_ylim(0, 1.)
            #~ ax.set_xlim(self.xmin, self.xmax)
            #~ ax.set_xlabel('generation')
            #~ ax.set_ylabel('frequency')
            #~ ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01), prop=legend_font)
            #~ ax.grid()
            #~ plt.draw()                                   # necessary to avoid AssertionError
#~ 
#~ 
    #~ def update(self):
        #~ if XDATA[-1] > self.xmax:
            #~ self.xmax += 100
            #~ for ax in self.figaxes:
                #~ ax.set_xlim(self.xmin, self.xmax)
        #~ self.draw()
        #~ 
    #~ def draw(self):
        #~ self.canvas.draw()
        #~ for i,pop in enumerate(self.populations):
            #~ ax = self.figaxes[i]
            #~ background = self.canvas.copy_from_bbox(ax.bbox)
            #~ self.canvas.restore_region(background)
            #~ for j,al in enumerate(self.labels[pop]):
                #~ line = self.axlines[i][j]
                #~ line.set_data(XDATA,YDATA[i][j])
                #~ try:
                    #~ ax.draw_artist(line)
                #~ except AssertionError:
                    #~ return
            #~ self.canvas.blit(ax.bbox)
        #~ return True
#~ 
    #~ def finalize(self):
        #~ for ax in self.figaxes:
            #~ background = self.canvas.copy_from_bbox(ax.bbox)
            #~ self.canvas.restore_region(background)
            #~ self.xmax = XDATA[-1]
            #~ ax.set_xlim(self.xmin, self.xmax)
        #~ self.canvas.draw()
        #~ 
    #~ def export_fig(self):
        #~ return self.fig
    #~ 
    #~ def store_data(self, filename='chart_data.npz'):
        #~ np.savez(filename, xdata=XDATA, ydata=YDATA)
    #~ 
    #~ def load_data(self, filename='chart_data.npz'):
        #~ global XDATA, YDATA
        #~ npz = np.load(filename)
        #~ XDATA = npz['xdata']
        #~ YDATA = npz['ydata']
        #~ self.xmin, self.xmax = XDATA[0], XDATA[-1]
        #~ for ax in self.figaxes:
                #~ ax.set_xlim(self.xmin, self.xmax)
