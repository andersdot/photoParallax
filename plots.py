import toy
import os
import testXD
from astroML.plotting import setup_text_plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import corner
from xdgmm import XDGMM
import drawEllipse
import comparePrior
import scipy.interpolate
import astropy.visualization as av

def figsize_and_margins(plotsize,subplots=(1,1),**absolute_margins):
    '''Determine figure size and margins from plot size and absolute margins

       Parameters:
         plotsize: (width, height) of plot area
         subplots: (nrows, ncols) of subplots
         left, right, top, bottom: absolute margins around plot area
         wspace, hspace: width and height spacing between subplots
       Returns:
         size: figure size for figsize argument of figure()
         margins: relative margins dict suitable for subplots_adjust()

       Example: making 2x2 grid of 3" square plots with specific spacings:

       sz, rm = figsize_and_margins((3,3), (2,2), left=1, right=.5,
                                                  top=.5, bottom=1,
                                                  wspace=.5, hspace=.5)
       figure(figsize=sz)
       subplots_adjust(**rm)
       subplot(221); subplot(222)
       subplot(223); subplot(224)
    '''
    #from matplotlib import rcParams
    pw,ph = plotsize
    nr,nc = subplots
    amarg = absolute_margins
    #dictionary for relative margins
    # initialize from rcParams with margins not in amarg
    rmarg = dict((m, mpl.rcParams['figure.subplot.' + m])
                for m in ('left','right','top','bottom','wspace','hspace')
                if m not in amarg
            )
    #subplots_adjust wants wspace and hspace relative to plotsize:
    if 'wspace' in amarg: rmarg['wspace'] = float(amarg['wspace']) / pw
    if 'hspace' in amarg: rmarg['hspace'] = float(amarg['hspace']) / ph
    #in terms of the relative margins:
    #width  * (right - left)
    #    = ncols * plot_width  + (ncols - 1) * wspace * plot_width
    #height * (top - bottom)
    #    = nrows * plot_height + (nrows - 1) * hspace * plot_height
    #solve for width and height, using absolute margins as necessary:
    #print(nc, rmarg['wspace'], pw, amarg.get('lef', 0), amarg.get('right', 0), rmarg.get('right', 1), rmarg.get('left', 0))
    width  = float((nc + (nc - 1) * rmarg['wspace']) * pw        \
                   + amarg.get('left',0) + amarg.get('right',0)) \
             / (rmarg.get('right',1) - rmarg.get('left',0))

    height = float((nr + (nr - 1) * rmarg['hspace']) * ph        \
                   + amarg.get('top',0) + amarg.get('bottom',0)) \
             / (rmarg.get('top',1) - rmarg.get('bottom',0))

    #now we can get any remaining relative margins
    if 'left'   in amarg: rmarg['left']   =     float(amarg['left'])   / width
    if 'right'  in amarg: rmarg['right']  = 1 - float(amarg['right'])  / width
    if 'top'    in amarg: rmarg['top']    = 1 - float(amarg['top'])    / height
    if 'bottom' in amarg: rmarg['bottom'] =     float(amarg['bottom']) / height
    #return figure size and relative margins
    return (width, height), rmarg

#Example usage: make 2 side-by-side 3" square figures
#from pylab import *
#fsize, margins = figsize_and_margins(plotsize=(3,3),subplots=(1,2))
#figure('My figure', figsize=fsize)
#adjust_subplots(**margins)
#subplot(121)
# ... plot something ...
#subplot(122)
# ... plot something ...
#show()


def gaussian(mean, sigma, array, amplitude=1.0):
    return amplitude/np.sqrt(2.*np.pi*sigma**2.)*np.exp(-(array - mean)**2./(2.*sigma**2.))


def plotPrior(xdgmm, ax, c='black', lw=1, stretch=False):
    for gg in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        if stretch:
            Stretch = av.PowerStretch(1./5)
            alpha=np.power(xdgmm.weights[gg]/np.max(xdgmm.weights), 1./10)
        else: alpha=xdgmm.weights[gg]/np.max(xdgmm.weights)
        ax.plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=lw, alpha=alpha)

def makeFigureInstance(x=1, y=1, left=1.0, right=0.25, top=0.25, bottom=0.75, wspace=0.5, hspace=0.5, figureSize=(3,3)):#, figsize=None, fontsize=12):
    sz, rm = figsize_and_margins(figureSize, (y,x), left=left, right=right,
                                           top=top, bottom=bottom,
                                           wspace=wspace, hspace=hspace)
    fig, ax = plt.subplots(y, x, figsize=sz)#, figsize=figsize)
    if (x > 1) or (y > 1): ax = ax.flatten()

    fig.subplots_adjust(**rm)
    #fig.subplots_adjust(left=0.1, right=0.9,
    #                    bottom=0.1, top=0.9,
    #                    wspace=0.4, hspace=0.5)
    #setup_text_plots(fontsize=fontsize, usetex=True)
    return fig, ax


def plot_samples(x, y, xerr, yerr, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor='black', titles=None, xlim=(0,1), ylim=(0,1), xlabel=None, ylabel=None, prior=False, xdgmm=None, pdf=False):#, annotateTextSize=18, figsize2x1 = (12, 5.5)):
    #setup_text_plots(fontsize=, usetex=True)
    plt.clf()
    alpha = 0.1
    alpha_points = 0.01
    fig, ax = makeFigureInstance(x=2, y=1)#, figsize=figsize2x1)
    #fig = plt.figure(figsize=figsize2x1)
    #fig.subplots_adjust(left=0.1, right=0.95,
    #                        bottom=0.15, top=0.95,
    #                        wspace=0.1, hspace=0.1)
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
    im = corner.hist2d(x, y, ax=ax[0], levels=levels, bins=200, no_fill_contours=True, plot_density=False, color=contourColor, rasterized=True, plot_contours=plot_contours, plot_datapoints=False)
    ax[0].scatter(x, y, s=1, lw=0, c=dataColor, alpha=alpha, zorder=0, rasterized=True)
    if prior:
        plotPrior(xdgmm, ax[1], c=dataColor)
    else:
        ax[1].errorbar(x[ind], y[ind], xerr=xerr[ind], yerr=[yerr[0][ind], yerr[1][ind]], fmt="none", zorder=0, mew=0, ecolor=dataColor, alpha=0.5, elinewidth=0.5)

    for i, axis in enumerate(ax):
        axis.set_xlim(xlim)
        axis.set_ylim(ylim[0], ylim[1]*1.1)
        axis.text(0.05, 0.95, titles[i],
                   ha='left', va='top', transform=axis.transAxes)#, fontsize=annotateTextSize)
        axis.set_xlabel(xlabel)
        if i in [1]: #(1, 3):
            axis.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            axis.set_ylabel(ylabel)
    if pdf: fig.savefig('plot_sample.pdf', dpi=400)
    fig.savefig('plot_sample.png')
    plt.close(fig)


def absMagError(parallax, parallax_err, apparentMag, absMag):
    absMag_errPlus = testXD.absMagKinda2absMag((parallax + parallax_err)*10.**(0.2*apparentMag)) - absMag
    absMag_errMinus = absMag - testXD.absMagKinda2absMag((parallax - parallax_err)*10.**(0.2*apparentMag))
    return [absMag_errMinus, absMag_errPlus]

def sampleXDGMM(xdgmm, Nsamples):
    sample = xdgmm.sample(Nsamples)
    negParallax = sample[:,1] < 0
    nNegP = np.sum(negParallax)
    while nNegP > 0:
        sampleNew = xdgmm.sample(nNegP)
        sample[negParallax] = sampleNew
        negParallax = sample[:,1] < 0
        nNegP = np.sum(negParallax)
    samplex = sample[:,0]
    sampley = testXD.absMagKinda2absMag(sample[:,1])
    return samplex, sampley

def likePriorPost(color, absMagKinda, color_err, absMagKinda_err, apparentMagnitude, xdgmm, xparallaxMAS, ndim=2, nPosteriorPoints=1000, projectedDimension=1):
    meanData, covData = testXD.matrixize(color, absMagKinda, color_err, absMagKinda_err)
    meanPrior, covPrior = testXD.matrixize(color, absMagKinda, color_err, 1e5)
    meanData = meanData[0]
    covData = covData[0]
    meanPrior = meanPrior[0]
    covPrior = covPrior[0]
    xabsMagKinda = testXD.parallax2absMagKinda(xparallaxMAS, apparentMagnitude)
    xColor = np.linspace(-2, 4, nPosteriorPoints)
    allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints)
    allMeansPrior, allAmpsPrior, allCovsPrior, summedPriorAbsMagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanPrior, covPrior, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints)
    allMeansColor, allAmpsColor, allCovsColor, summedPosteriorColor = testXD.absMagKindaPosterior(xdgmm, ndim, meanData, covData, xColor, projectedDimension=0, nPosteriorPoints=1000, prior=False)
    posteriorParallax = summedPosteriorAbsmagKinda*10.**(0.2*apparentMagnitude)
    priorParallax = summedPriorAbsMagKinda*10.**(0.2*apparentMagnitude)
    likeParallax = gaussian(absMagKinda/10.**(0.2*apparentMagnitude), absMagKinda_err/10.**(0.2*apparentMagnitude), xparallaxMAS)
    return likeParallax, priorParallax, posteriorParallax, summedPosteriorColor

def main():
    #    for label, style in zip(['paper', 'talk'],['seaborn-paper', 'seaborn-talk']):
    pdf = True
    plot_data = False
    plot_dust = False
    plot_prior = False
    plot_m67 = False
    plot_compare = False
    plot_expectation = False
    plot_odd_examples = True
    plot_examples = False
    plot_delta = False
    plot_deltacdf = False
    plot_nobias = False
    plot_wtf = False
    plot_toy = False

    #figsize2x1 = (12, 5.5)
    #figsize2x2 = (12, 11)
    #figsize3x2 = (18, 11)
    style = 'seaborn-paper'
    #plt.style.use(style)
    #fontsize = 12
    #annotateTextSize = 12
    #legendTextSize = 12
    params = {
        'axes.labelsize' : 9,
        'font.size' : 9,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5]
        }
    mpl.rcParams.update(params)
    #mpl.rcParams['xtick.labelsize'] = fontsize
    #mpl.rcParams['ytick.labelsize'] = fontsize
    #mpl.rcParams['axes.labelsize'] = fontsize
    #mpl.rcParams['font.size'] = fontsize
    nsubsamples = 1024
    np.random.seed(0)

    trueColor='darkred'
    priorColor= '#6baed6' #'#9ebcda' #'#9ecae1' #'royalblue'
    cmap_prior = 'Blues'
    posteriorColor= '#984ea3' #'#7a0177' #'#8856a7' #'#810f7c' #'#08519c' #'darkblue'
    dataColor='black'
    posteriorMapColor = 'Blues'

    mag1 = 'J'
    mag2 = 'K'
    absmag = 'J'
    xlabel_cmd = r'$(J-K_s)^C$'
    ylabel_cmd = r'$M_J^C$'
    xlim_cmd = [-0.25, 1.25]
    ylim_cmd = [6, -6]

    dustFile = 'dustCorrection.128gauss.dQ0.05.10th.2MASS.All.npz'
    xdgmmFile = 'xdgmm.128gauss.dQ0.05.10th.2MASS.All.npz.fit'
    posteriorFile = 'posteriorParallax.128gauss.dQ0.05.10th.2MASS.All.npz'

    xdgmm = XDGMM(filename=xdgmmFile)
    #generate toy model plot
    mtrue=-1.37
    btrue=0.2
    ttrue=0.8
    nexamples=5
    if plot_toy:
        fig, ax = makeFigureInstance(x=2, y=2, wspace=0.75)
        toy.makeplots(mtrue=mtrue, btrue=btrue, ttrue=ttrue, nexamples=nexamples,
        trueColor=trueColor, priorColor=priorColor, posteriorColor=posteriorColor, dataColor=dataColor, posteriorMapColor=posteriorMapColor, fig=fig, axes=ax)
        os.rename('toy.paper.pdf', 'paper/toy.pdf')
    #----------------------------------------------


    #generate raw data plot
    tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()
    posterior = np.load(posteriorFile)
    mean = posterior['mean']
    sigma = np.sqrt(posterior['var'])
    positive = (tgas['parallax'] > 0.) & (mean > 0.)
    ind = np.random.randint(0, len(tgas[positive]), nsubsamples)

    dustEBV = 0.0
    absMagKinda, apparentMagnitude = testXD.absMagKindaArray(absmag, dustEBV, bandDictionary, tgas['parallax'])
    absMagKinda_err = tgas['parallax_error']*10.**(0.2*apparentMagnitude)
    color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)[positive]
    color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)[positive]
    absMag = testXD.absMagKinda2absMag(tgas['parallax'][positive]*10.**(0.2*apparentMagnitude[positive]))
    absMag_err = absMagError(tgas['parallax'][positive], tgas['parallax_error'][positive], apparentMagnitude[positive], absMag)
    titles = ["Observed Distribution", "Obs+Noise Distribution"]

    if plot_data:
        plot_samples(color, absMag, color_err, absMag_err, ind, contourColor='grey', rasterized=True, plot_contours=True, dataColor=dataColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, pdf=pdf) #, annotateTextSize=annotateTextSize, figsize2x1=figsize2x1)
        if pdf: os.rename('plot_sample.pdf', 'paper/data.pdf')
        os.rename('plot_sample.png', 'data.png')

    #color_raw = color
    #color_err_raw = color_err
    #absMag_raw = absMag
    #absMag_err_raw = absMag_err
    #absMagKinda_raw = absMagKinda
    #absMagKinda_err_raw = absMagKinda_err
    #-------------------------------------------------------


    #dust plot
    if plot_dust:
        fig, ax = makeFigureInstance(figureSize = (6,3),left=0.75)
        comparePrior.dustViz(ngauss=128, quantile=0.05, iter='10th', survey='2MASS', dataFilename='All.npz', ax=ax, tgas=tgas)
        fig.savefig('paper/dust.pdf', dpi=400)
        fig.savefig('dust.png')
        plt.close(fig)
    #-------------------------------------------------------


    #generate prior plot
    if plot_prior:
        samplex, sampley = sampleXDGMM(xdgmm, len(tgas))
        titles = ["Extreme Deconvolution\n  resampling", "Extreme Deconvolution\n  cluster locations"]
        plot_samples(samplex, sampley, None, None, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor=priorColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, prior=True, xdgmm=xdgmm, pdf=pdf) #, annotateTextSize=annotateTextSize, figsize2x1=figsize2x1)
        if pdf: os.rename('plot_sample.pdf', 'paper/prior.pdf')
        os.rename('plot_sample.png', 'prior.png')
    #-------------------------------------------------------


    data = np.load(dustFile)
    dustEBV = data['ebv']
    absMagKinda, apparentMagnitude = testXD.absMagKindaArray(absmag, dustEBV, bandDictionary, tgas['parallax'])
    color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)
    color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
    absMagKinda_err = tgas['parallax_error']*10.**(0.2*apparentMagnitude)
    #-------------------------------------------------------


    #M67 plot
    if plot_m67:
        fig, ax = makeFigureInstance(x=2, y=2, hspace=1.0, wspace=1.0)
        #setup_text_plots(fontsize=fontsize, usetex=True)
        #fig, ax = plt.subplots(2,2, figsize=figsize2x2)
        #fig.subplots_adjust(left=0.1, right=0.95,
        #                               bottom=0.1, top=0.95,
        #                            wspace=0.25, hspace=0.25)

        #ax = ax.flatten()
        nPosteriorPoints = 1000
        print(dataColor)
        #def distanceTest(tgas, xdgmm, nPosteriorPoints, data1, data2, err1, err2, xlim, ylim, plot2DPost=False, dataColor='black', priorColor='green', truthColor='red', posteriorColor='blue', dl=0.1, db=0.1):

        testXD.distanceTest(tgas, xdgmm, nPosteriorPoints, color, absMagKinda, color_err, absMagKinda_err, xlim_cmd, ylim_cmd, bandDictionary, absmag, dataColor=dataColor, priorColor=priorColor, truthColor=trueColor, posteriorColor=posteriorColor, figDist=fig, axDist=ax, xlabel=xlabel_cmd, ylabel=ylabel_cmd, dl=0.075, db=0.075)
        plt.tight_layout()
        if pdf: fig.savefig('paper/m67.pdf', dpi=400)
        fig.savefig('m67.png')
        plt.close(fig)
    #-------------------------------------------------------


    color = color[positive]
    color_err = color_err[positive]
    apparentMagnitude = apparentMagnitude[positive]
    absMagKinda_dust = absMagKinda[positive]
    absMagKinda_dust_err = absMagKinda_err[positive]
    absMag_dust = testXD.absMagKinda2absMag(absMagKinda[positive])
    absMag_dust_err = absMagError(tgas['parallax'][positive], tgas['parallax_error'][positive], apparentMagnitude, absMag_dust)


    #generate comparison prior plot
    if plot_compare:
        #setup_text_plots(fontsize=fontsize, usetex=True)
        plt.clf()
        alpha = 0.1
        alpha_points = 0.01
        fig, ax = makeFigureInstance(x=2, y=1)#, figsize=figsize2x1)
        #fig = plt.figure(figsize=figsize2x1)
        #fig.subplots_adjust(left=0.1, right=0.95,
        #                    bottom=0.15, top=0.95,
        #                    wspace=0.1, hspace=0.1)
        #ax1 = fig.add_subplot(121)
        #ax2 = fig.add_subplot(122)
        #ax = [ax1, ax2]
        titles =  ['Exp Dec Sp \nDen Prior', 'CMD Prior']
    for i, file in enumerate(['posteriorSimple.npz', posteriorFile]):
        data = np.load(file)
        posterior = data['posterior']
        sigma = np.sqrt(data['var'])
        mean = data['mean']
        absMag = testXD.absMagKinda2absMag(mean[positive]*10.**(0.2*apparentMagnitude))
        absMag_err = absMagError(mean[positive], sigma[positive], apparentMagnitude, absMag)
        if plot_compare:    #ax[i].scatter(color[ind], absMag[ind], c=posteriorColor, s=1, lw=0, alpha=alpha, zorder=0)
            ax[i].errorbar(color[ind], absMag[ind], xerr=color_err[ind], yerr=[absMag_err[0][ind], absMag_err[1][ind]], fmt="none", zorder=0, mew=0, ecolor=posteriorColor, alpha=0.5, elinewidth=0.5, color=posteriorColor)
            ax[i].set_xlim(xlim_cmd)
            ax[i].set_ylim(ylim_cmd[0], ylim_cmd[1]*1.1)
            ax[i].text(0.05, 0.95, titles[i], ha='left', va='top', transform=ax[i].transAxes) #, fontsize=annotateTextSize)
            ax[i].set_xlabel(xlabel_cmd)
            if i in [1]:
                ax[i].yaxis.set_major_formatter(plt.NullFormatter())
            else:
                ax[i].set_ylabel(ylabel_cmd)
    if plot_compare:
        if pdf: fig.savefig('paper/comparePrior.pdf', dpi=400)
        fig.savefig('comparePrior.png')
        plt.close(fig)
    #-------------------------------------------------------


    #generate expectation plot

    absMag = testXD.absMagKinda2absMag(mean[positive]*10.**(0.2*apparentMagnitude))
    absMag_err = absMagError(mean[positive], sigma[positive], apparentMagnitude, absMag)
    titles = ["De-noised Expectation \nValues", "Posterior Distributions"]
    if plot_expectation:
        plot_samples(color, absMag, color_err, absMag_err, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor=posteriorColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, pdf=pdf)#, annotateTextSize=annotateTextSize, figsize2x1=figsize2x1)
        if pdf: os.rename('plot_sample.pdf', 'paper/posteriorCMD.pdf')
        os.rename('plot_sample.png', 'posteriorCMD.png')
    #-------------------------------------------------------


    #posterior example plot
    if plot_examples:
        colorBins = [0.0, 0.2, 0.4, 0.7, 1.0]
        digit = np.digitize(color, colorBins)

        ndim = 2
        nPosteriorPoints = 1000 #number of elements in the posterior array
        projectedDimension = 1  #which dimension to project the prior onto
        xparallaxMAS = np.linspace(0, 10, nPosteriorPoints)


        #plot likelihood and posterior in each axes
        for iteration in np.arange(20, 40):
            fig, ax = makeFigureInstance(x=3, y=2, hspace=0.75, figureSize=(2,2)) #, figsize=figsize3x2)
            #fig, ax = plt.subplots(2, 3, figsize=figsize3x2)
            #ax = ax.flatten()
            #fig.subplots_adjust(left=0.1, right=0.9,
            #                        bottom=0.1, top=0.8,
            #                        wspace=0.4, hspace=0.5)


            plotPrior(xdgmm, ax[0], c=priorColor, lw=1)
            ax[0].set_xlim(xlim_cmd)
            ax[0].set_ylim(ylim_cmd)
            ax[0].set_xlabel(xlabel_cmd)
            ax[0].set_ylabel(ylabel_cmd)

            for i in range(np.max(digit)):
                currentInd = np.where((digit == i))[0]
                index = currentInd[np.random.randint(0, high=len(currentInd))]
                ax[0].scatter(color[index], absMag_dust[index], c=dataColor, s=20)
                ax[0].errorbar(color[index], absMag_dust[index], xerr=[[color_err[index], color_err[index]]], yerr=[[absMag_dust_err[0][index], absMag_dust_err[1][index]]], fmt="none", zorder=0, lw=2.0, mew=0, alpha=1.0, color=dataColor, ecolor=dataColor)
                ax[0].annotate(str(i+1), (color[index]+0.075, absMag_dust[index]+0.175))#, fontsize=annotateTextSize)
                #print len(color), len(absMagKinda_dust), len(color_err), len(absMagKinda_dust_err), len(apparentMagnitude)
                likeParallax, priorParallax, posteriorParallax, posteriorColor = likePriorPost(color[index], absMagKinda_dust[index], color_err[index], absMagKinda_dust_err[index], apparentMagnitude[index], xdgmm, xparallaxMAS, ndim=2, nPosteriorPoints=1000, projectedDimension=1)

                l1, = ax[i+1].plot(xparallaxMAS, likeParallax*np.max(posteriorParallax)/np.max(likeParallax), lw=1, color=dataColor, zorder=100)
                l2, = ax[i+1].plot(xparallaxMAS, priorParallax*np.max(posteriorParallax)/np.max(priorParallax), lw=0.5, color=priorColor)
                l3, = ax[i+1].plot(xparallaxMAS, posteriorParallax, lw=2, color=posteriorColor)
                maxInd = posteriorParallax == np.max(posteriorParallax)
                maxPar = xparallaxMAS[maxInd]
                maxY = posteriorParallax[maxInd]
                if maxPar < 5: annX = 9
                else: annX = 0
                if i == 1: annY = 0.75*maxY
                else: annY = maxY/1.1
                ax[i+1].text(annX, annY, str(i+1))
                ax[i+1].set_xlabel(r'$\varpi$ [mas]')
                ax[i+1].tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                if i+1 == 1:
                    leg = fig.legend((l1, l2, l3), ('likelihood', 'prior', 'posterior'), 'upper right') #, fontsize=legendTextSize)
                    leg.get_frame().set_alpha(1.0)
                #plt.tight_layout()
                if pdf: fig.savefig('posterior_' + str(iteration) + '.pdf', dpi=400)
            fig.savefig('paper/posterior.pdf', dpi=400)
            fig.tight_layout()
            fig.savefig('posterior.png')
            plt.close(fig)

    #-------------------------------------------------------
    #odd posterior example plot
    if plot_odd_examples:


        #choose indices for odd plot_examples
        #odd colors and magnitudes
        #come back and do parallax negative
        SN = tgas['parallax'][positive]/tgas['parallax_error'][positive]
        oddIndicesWD_LowSN  = np.where(np.logical_and((absMag_dust > 6.*color + 5.), (SN <= 5)))[0]
        oddIndicesWD_HighSN = np.where(np.logical_and((absMag_dust > 6.*color + 5.), (SN > 5)))[0]#3.6)[0]
        oddIndicesSSG = np.where(np.logical_and((absMag_dust < 7.5*color - 1.5), (absMag_dust > -8.1*color + 7.8)))[0]
        oddIndicesPN_LowSN   = np.where(np.logical_and(SN <= 5, np.logical_and((absMag_dust < 7.5*color - 4.25),(absMag_dust < -4.75*color - 0.6))))[0]
        oddIndicesPN_HighSN  = np.where(np.logical_and(SN > 5, np.logical_and((absMag_dust < 7.5*color - 4.25),(absMag_dust < -4.75*color - 0.6))))[0]

        ndim = 2
        nPosteriorPoints = 1000 #number of elements in the posterior array
        projectedDimension = 1  #which dimension to project the prior onto
        xparallaxMAS = np.linspace(0, 10, nPosteriorPoints)
        xarray = np.logspace(-2, 2, 1000)
        xColor = np.linspace(-2, 4, nPosteriorPoints)
        samplex, sampley = sampleXDGMM(xdgmm, len(tgas)*10)
        #plot likelihood and posterior in each axes
        for iteration in np.arange(0, 10):
            fig, ax = makeFigureInstance(x=3, y=2, hspace=0.75, figureSize=(2,2)) #, figsize=figsize3x2)
            #fig, ax = plt.subplots(2, 3, figsize=figsize3x2)
            #ax = ax.flatten()
            #fig.subplots_adjust(left=0.1, right=0.9,
            #                        bottom=0.1, top=0.8,
            #                        wspace=0.4, hspace=0.5)
            ax[0].hist2d(samplex, sampley, bins=500, norm=mpl.colors.LogNorm(), cmap=plt.get_cmap(cmap_prior), zorder=-1)
            #plotPrior(xdgmm, ax[0], c=priorColor, lw=1, stretch=True)
            ax[0].set_ylim(15, -10)
            ax[0].set_xlim(-1.2, 2)
            ax[0].set_ylim(ylim_cmd[0]+3, ylim_cmd[1]-3)
            ax[0].set_xlabel(xlabel_cmd)
            ax[0].set_ylabel(ylabel_cmd)

            for i, indices in enumerate([oddIndicesWD_LowSN, oddIndicesWD_HighSN, oddIndicesSSG, oddIndicesPN_LowSN, oddIndicesPN_HighSN]):
                print(len(indices), indices)
                #if i == 0: index = indices[iteration]
                #else: index = indices[np.random.randint(0, high=len(indices))]
                index = indices[np.random.randint(0, high=len(indices))]
                ax[0].scatter(color[index], absMag_dust[index], c=dataColor, s=20)
                yplus = absMag_dust_err[0][index]
                yminus = absMag_dust_err[1][index]
                if np.isnan(yplus): yplus = 10.
                if np.isnan(yminus): yminus = 10.
                print(yplus, yminus)
                ax[0].errorbar(color[index], absMag_dust[index], xerr=[[color_err[index]], [color_err[index]]], yerr=[[yplus], [yminus]], fmt="none", zorder=0, lw=2.0, mew=0, alpha=1.0, color=dataColor, ecolor=dataColor)
                ax[0].annotate(str(i+1), (color[index]+0.075, absMag_dust[index]+0.175))#, fontsize=annotateTextSize)
                #print len(color), len(absMagKinda_dust), len(color_err), len(absMagKinda_dust_err), len(apparentMagnitude)
                likeParallax, priorParallax, posteriorParallax, posteriorColorArray = likePriorPost(color[index], absMagKinda_dust[index], color_err[index], absMagKinda_dust_err[index], apparentMagnitude[index], xdgmm, xparallaxMAS, ndim=2, nPosteriorPoints=1000, projectedDimension=1)

                likeParallaxFull, priorParallaxFull, posteriorParallaxFull, posteriorColorFull = likePriorPost(color[index], absMagKinda_dust[index], color_err[index], absMagKinda_dust_err[index], apparentMagnitude[index], xdgmm, xarray, ndim=2, nPosteriorPoints=1000, projectedDimension=1)

                meanPosteriorParallax = scipy.integrate.cumtrapz(posteriorParallaxFull*xarray, x=xarray)[-1]
                x2PosteriorParallax = scipy.integrate.cumtrapz(posteriorParallaxFull*xarray**2., x=xarray)[-1]
                varPosteriorParallax = x2PosteriorParallax - meanPosteriorParallax**2.
                meanPosteriorColor = scipy.integrate.cumtrapz(posteriorColorFull*xColor, x=xColor)[-1]
                x2PosteriorColor = scipy.integrate.cumtrapz(posteriorColorFull*xColor**2., x=xColor)[-1]
                varPosteriorColor = x2PosteriorColor - meanPosteriorColor**2.

                absMagPost = testXD.absMagKinda2absMag(meanPosteriorParallax*10.**(0.2*apparentMagnitude[index]))
                absMag_errPost = absMagError(meanPosteriorParallax, np.sqrt(varPosteriorParallax), apparentMagnitude[index], absMagPost)
                yplus = absMag_dust_err[0][index]
                yminus = absMag_dust_err[1][index]

                if np.isnan(yplus): yplus = 10.
                if np.isnan(yminus): yminus = 10.


                l1, = ax[i+1].plot(xparallaxMAS, likeParallax*np.max(posteriorParallax)/np.max(likeParallax), lw=2, color=dataColor, zorder=100)
                l2, = ax[i+1].plot(xparallaxMAS, priorParallax*np.max(posteriorParallax)/np.max(priorParallax), lw=2, color=priorColor, linestyle='--')
                l3, = ax[i+1].plot(xparallaxMAS, posteriorParallax, lw=2, color=posteriorColor)
                ax[0].scatter(meanPosteriorColor, absMagPost, c=posteriorColor, s=20)
                ax[0].errorbar(meanPosteriorColor, absMagPost, xerr=[[np.sqrt(varPosteriorColor)], [np.sqrt(varPosteriorColor)]], yerr=[[yplus], [yminus]], fmt="none", zorder=0, lw=2.0, mew=0, alpha=1.0, color=posteriorColor, ecolor=posteriorColor)
                maxInd = np.where(posteriorParallax == np.max(posteriorParallax))[0]
                maxPar = xparallaxMAS[maxInd]
                maxY = posteriorParallax[maxInd]
                if maxPar < 5: annX = 9
                else: annX = 0
                if i == 1: annY = 0.75*maxY
                else: annY = maxY/1.1
                ax[i+1].text(annX, annY, str(i+1))
                ax[i+1].set_xlabel(r'$\varpi$ [mas]')
                ax[i+1].tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                if i+1 == 1:
                    leg = fig.legend((l1, l2, l3), ('likelihood', 'prior', 'posterior'), 'upper right') #, fontsize=legendTextSize)
                    leg.get_frame().set_alpha(1.0)
                #plt.tight_layout()
                if pdf: fig.savefig('posterior_' + str(iteration) + '_odd.pdf', dpi=400)
            fig.savefig('paper/posterior_odd.pdf', dpi=400)
            fig.tight_layout()
            fig.savefig('posterior_odd.png')
            plt.close(fig)
    #-------------------------------------



    #delta plot
    label = r'$\mathrm{ln} \, \tilde{\sigma}_{\varpi}^2 - \mathrm{ln} \, \sigma_{\varpi}^2$'
    contourColor = '#1f77b4'
    color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)
    color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)

    x = color
    y = np.log(sigma**2.) - np.log(tgas['parallax_error']**2.)
    colorDeltaVar = y
    notnans = ~np.isnan(sigma) & ~np.isnan(tgas['parallax_error']) & ~np.isnan(color)

    if plot_delta:
        fig, ax = makeFigureInstance(x=2, y=1, wspace=1.0) # , figsize=figsize2x1)
        #fig, ax = plt.subplots(1, 2, figsize=figsize2x1)
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
        norm = plt.matplotlib.colors.Normalize(vmin=-1.5, vmax=1)
        cmap = 'inferno'
        ax[0].scatter(x[notnans], y[notnans], c=y[notnans], s=1, lw=0, alpha=0.05, norm=norm, cmap=cmap, rasterized=True)
        #corner.hist2d(x[notnans], y[notnans], bins=200, ax=ax[0], levels=levels, no_fill_contours=True, plot_density=False, plot_data=False, color=contourColor, rasterized=True)
        ax[0].set_xlabel(xlabel_cmd)
        ax[0].set_ylim(-6, 2)
        ax[0].set_xlim(-0.5, 2)
        ax[0].set_ylabel(label)
        cNorm  = plt.matplotlib.colors.Normalize(vmin=0.1, vmax=2)
        ax[1].scatter(x[positive], absMag, s=1, lw=0, c=y[positive], alpha=0.05, norm=norm, cmap=cmap, rasterized=True)
        ax[1].set_xlim(xlim_cmd)
        ax[1].set_ylim(ylim_cmd)
        ax[1].set_xlabel(xlabel_cmd)
        ax[1].set_ylabel(ylabel_cmd)
        if pdf: fig.savefig('paper/delta.pdf', dpi=400)
        fig.savefig('delta.png')
        plt.close(fig)

    #delta cdf plot
    ratioCmd = sigma[notnans]**2./tgas['parallax_error'][notnans]**2.
    lnratio = np.log(ratioCmd)

    if plot_deltacdf:
        plt.clf()
        fig, ax = makeFigureInstance(left=0.75)
        N = len(lnratio)
        ys = np.arange(0+0.5/N, 1, 1.0/N)
        sinds = np.argsort(lnratio)
        f = scipy.interpolate.interp1d(lnratio[sinds], ys)
        f_inv = scipy.interpolate.interp1d(ys, lnratio[sinds])
        ax.plot(lnratio[sinds], ys, 'k-', lw=2)
        fac2 = np.log(1/4.)
        fac1 = 0.
        ax.plot([fac2, fac2],[-1, f(fac2)], 'k--', lw=2)
        ax.plot([-6, fac2],[f(fac2), f(fac2)], 'k--' ,lw=2)
        ax.plot([fac1, fac1], [-1, f(fac1)], 'k--', lw=2)
        ax.plot([-6, fac1], [f(fac1), f(fac1)], 'k--', lw=2)
        ax.plot([f_inv(0.5), f_inv(0.5)], [-1, 0.5], 'k--', lw=2)
        ax.plot([-6, f_inv(0.5)], [0.5, 0.5], 'k--', lw=2)
        ax.set_xlabel(label)
        ax.set_ylabel('cumulative fraction')
        ax.set_xlim(-6, 2)
        ax.set_ylim(-0.05, 1.05)
        if pdf: fig.savefig('paper/deltaCDF.pdf', dpi=400)
        fig.savefig('deltaCDF.png')
        plt.close(fig)
        print('fraction of stars which decreased in variance: ', f(fac1))
    #delta mean vs gaia uncertainty
    y = mean - tgas['parallax']
    x = tgas['parallax_error']
    good = ~np.isnan(y) & ~np.isnan(x)

    if plot_nobias:
        plt.clf()
        fig, ax = makeFigureInstance(left=0.75)
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
        contourColor = '#1f77b4'
        contourColor = 'black'
        #corner.hist2d(x[good], y[good], bins=200, ax=ax, levels=levels, no_fill_contours=True, plot_density=False, plot_data=False, color=contourColor, rasterized=True)
        #norm = plt.matplotlib.colors.Normalize(vmin=0.0, vmax=1)
        ax.scatter(x[notnans], y[notnans], c=colorDeltaVar[notnans], s=1, lw=0, alpha=0.05, norm=norm, cmap=cmap, rasterized=True)
        #ax.scatter(x[good], y[good], c=sigma[good], s=1, lw=0, alpha=0.05, norm=norm, cmap=cmap, rasterized=True)
        #ax.scatter(x[good], y[good], c=np.sqrt(sigma[good]), s=1, rasterized=True, zorder=0, alpha=0.1, cmap=cmap, norm=norm)
        ax.plot([0, 1.1], [0,0], 'k--', lw=1)
        ax.set_xlim(0.15, 1.05)
        ax.set_ylim(-2.5, 2.5)
        ylabel = r'$\mathrm{Posterior \, Expectation \, Value} - \varpi_n$'
        xlabel = r'$\sigma_{\varpi,n}$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if pdf: fig.savefig('paper/deltaParallax.pdf', dpi=400)
        fig.savefig('deltaParallax.png')
        plt.close(fig)

    #what's that feature plot
    if plot_wtf:
        fig, ax = makeFigureInstance(left=0.75)
        ax.scatter(color[positive], absMag, s=1, lw=0, c=dataColor, alpha=0.01, zorder=0, rasterized=True)
        ax.set_xlim(xlim_cmd)
        ax.set_ylim(ylim_cmd)
        ax.set_xlabel(xlabel_cmd)
        ax.set_ylabel(ylabel_cmd)
        lowerMainSequence = (0.4, 5.5)
        upperMainSequence = (-0.225, 2)
        binarySequence = (0.65, 4)
        redClump = (0.35, -2)
        redGiantBranch = (1.0, -2)
        turnOff = (-0.15, 3.5)
        features = [lowerMainSequence, upperMainSequence, binarySequence, redClump, redGiantBranch, turnOff]
        labels = ['lower MS', 'upper MS', 'binary sequence', 'red clump', 'RGB', 'MS turn off', 'subgiant branch']
        for l, f in zip(labels, features): ax.text(f[0], f[1], l) #, fontsize=annotateTextSize)
        if pdf: fig.savefig('paper/whatsThatFeature.pdf', dpi=400)
        fig.savefig('whatsThatFeature.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
