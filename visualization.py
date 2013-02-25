import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# n_loci: (nx, ny)    -->  subplot(ny,nx,i+1) for i in range (n_loci)
loc_dims = {'1': (1,1), \
            '2': (1,2), \
            '3': (2,2), \
            '4': (2,2), \
            '5': (2,3), \
            '6': (2,3), \
            '7': (2,4), \
            '8': (2,4), \
            '9': (3,3)
           }

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
