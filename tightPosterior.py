import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import comparePrior
import testXD
from xdgmm import XDGMM
import testXD
import stellarTwins as st
np.random.seed(42)

def distanceFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'distanceQuantiles.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename

def dustFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename

tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()

ngauss = 128
survey = '2MASS'
quantile = 0.05
dataFilename = 'All.npz'
norm = mpl.colors.Normalize(vmin=-1, vmax=5)
iter = '8th'

if survey == 'APASS':
    mag1 = 'B'
    mag2 = 'V'
    absmag = 'G'
    xlabel='B-V'
    ylabel = r'M$_\mathrm{G}$'
    xlim = [-0.2, 2]
    ylim = [9, -2]

if survey == '2MASS':
    mag1 = 'J'
    mag2 = 'K'
    absmag = 'J'
    xlabel = 'J-K$_s$'
    ylabel = r'M$_\mathrm{J}$'
    xlim = [-0.25, 1.25]
    ylim = [6, -6]



tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()

dustFile = dustFilename(ngauss, quantile, iter, survey, dataFilename)
data = np.load(dustFile)
dustEBV = data['ebv']

color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)
absMagKinda, apparentMagnitude = testXD.absMagKindaArray(absmag, dustEBV, bandDictionary, tgas['parallax'])

color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
absMagKinda_err = tgas['parallax_error']*10.**(0.2*bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])



colorBins = [0.0, 0.2, 0.4, 0.7, 1.0]
digit = np.digitize(color, colorBins)
debug = False
ndim = 2
nPosteriorPoints = 1000 #number of elements in the posterior array
projectedDimension = 1  #which dimension to project the prior onto
ndim = 2
xparallaxMAS = np.linspace(0, 10, nPosteriorPoints)


y = absMagKinda
yplus  = y + absMagKinda_err
yminus = y - absMagKinda_err
parallaxErrGoesNegative = yminus < 0
absMagYMinus = testXD.absMagKinda2absMag(yminus)
absMagYMinus[parallaxErrGoesNegative] = -50.
yerr_minus = testXD.absMagKinda2absMag(y) - absMagYMinus
yerr_plus = testXD.absMagKinda2absMag(yplus) - testXD.absMagKinda2absMag(y)


#plot likelihood and posterior in each axes
for iteration in np.arange(20, 40):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.flatten()
    fig.subplots_adjust(left=0.1, right=0.9,
                                bottom=0.1, top=0.9,
                                wspace=0.4, hspace=0.5)

    #plot prior in upper left
    xdgmmFilename = 'xdgmm.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.2MASS.All.npz.fit'
    xdgmm = XDGMM(filename=xdgmmFilename)
    testXD.plotPrior(xdgmm, ax[0], c='k', lw=1)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xlabel('$(J-K)^C$', fontsize=18)
    ax[0].set_ylabel('$M_J^C$', fontsize=18)

    for i in range(np.max(digit)):
        currentInd = np.where((digit == i))[0]
        index = currentInd[np.random.randint(0, high=len(currentInd))]

        print 'yerr minus: ' + str(yerr_minus[index]) + ' yerr plus: ' + str(yerr_plus[index])
        ax[0].scatter(color[index], testXD.absMagKinda2absMag(absMagKinda[index]), c='black')
        ax[0].errorbar(color[index], testXD.absMagKinda2absMag(absMagKinda[index]), xerr=[[color_err[index], color_err[index]]], yerr=[[yerr_minus[index], yerr_plus[index]]], fmt="none", zorder=0, lw=2.0, mew=0, alpha=1.0, color='black', ecolor='black')
        ax[0].annotate(str(i+1), (color[index]+0.05, testXD.absMagKinda2absMag(absMagKinda[index])+0.15), fontsize=18)
        meanData, covData = testXD.matrixize(color[index], absMagKinda[index], color_err[index], absMagKinda_err[index])
        meanPrior, covPrior = testXD.matrixize(color[index], absMagKinda[index], color_err[index], 1e5)
        meanData = meanData[0]
        covData = covData[0]
        meanPrior = meanPrior[0]
        covPrior = covPrior[0]
        xabsMagKinda = testXD.parallax2absMagKinda(xparallaxMAS, apparentMagnitude[index])

        if debug:
            windowFactor = 15. #the number of sigma to sample in mas for plotting
            minParallaxMAS = tgas['parallax'][index] - windowFactor*tgas['parallax_error'][index]
            maxParallaxMAS = tgas['parallax'][index] + windowFactor*tgas['parallax_error'][index]
            xparallaxMAS, xabsMagKinda = testXD.plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude[index], nPosteriorPoints=nPosteriorPoints)
            xabsMagKinda = xabsMagKinda[::-1]
            xparallaxMAS = xparallaxMAS[::-1]
            positive = xparallaxMAS > 0.
            if np.sum(positive) == 0:
                print str(index) + ' has no positive distance values'
                continue
            logDistance = np.log10(1./xparallaxMAS[positive])
        allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints)
        allMeansPrior, allAmpsPrior, allCovsPrior, summedPriorAbsMagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanPrior, covPrior, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints)
        print np.min(summedPriorAbsMagKinda), np.max(summedPriorAbsMagKinda)
        posteriorParallax = summedPosteriorAbsmagKinda*10.**(0.2*apparentMagnitude[index])
        priorParallax = summedPriorAbsMagKinda*10.**(0.2*apparentMagnitude[index])
        likeParallax = st.gaussian(absMagKinda[index]/10.**(0.2*apparentMagnitude[index]), absMagKinda_err[index]/10.**(0.2*apparentMagnitude[index]), xparallaxMAS)

        l1, = ax[i+1].plot(xparallaxMAS, likeParallax*np.max(posteriorParallax)/np.max(likeParallax), alpha=0.5, lw=2, color='black')
        l2, = ax[i+1].plot(xparallaxMAS, priorParallax*np.max(posteriorParallax)/np.max(priorParallax), lw=0.5, color='black')
        l3, = ax[i+1].plot(xparallaxMAS, posteriorParallax, lw=2, color='black')
        ax[i+1].set_title(str(i+1))
        ax[i+1].set_xlabel(r'$\varpi$ [mas]', fontsize=18)
        ax[i+1].tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        if i+1 == 1: fig.legend((l1, l2, l3), ('likelihood', 'prior', 'posterior'), 'upper right', fontsize=15)
        #plt.tight_layout()
        fig.savefig('tightPosterior_' + str(iteration) + '.png')
        plt.close(fig)
