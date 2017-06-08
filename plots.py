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

def gaussian(mean, sigma, array, amplitude=1.0):
    return amplitude/np.sqrt(2.*np.pi*sigma**2.)*np.exp(-(array - mean)**2./(2.*sigma**2.))


def plotPrior(xdgmm, ax, c='black', lw=1):
    for gg in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        ax.plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=lw, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))


def plot_samples(x, y, xerr, yerr, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor='black', titles=None, xlim=(0,1), ylim=(0,1), xlabel=None, ylabel=None, prior=False, xdgmm=None, pdf=False):
    setup_text_plots(fontsize=16, usetex=True)
    plt.clf()
    alpha = 0.1
    alpha_points = 0.01
    fig = plt.figure(figsize=(12, 5.5))
    fig.subplots_adjust(left=0.1, right=0.95,
                            bottom=0.15, top=0.95,
                            wspace=0.1, hspace=0.1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
    im = corner.hist2d(x, y, ax=ax1, levels=levels, bins=200, no_fill_contours=True, plot_density=False, color=contourColor, rasterized=True, plot_contours=plot_contours, plot_datapoints=False)
    ax1.scatter(x, y, s=1, lw=0, c=dataColor, alpha=alpha, zorder=0, rasterized=True)
    if prior:
        plotPrior(xdgmm, ax2, c=dataColor)
    else:
        ax2.errorbar(x[ind], y[ind], xerr=xerr[ind], yerr=[yerr[0][ind], yerr[1][ind]], fmt="none", zorder=0, mew=0, ecolor=dataColor, alpha=0.5, elinewidth=0.5)
    ax = [ax1, ax2]
    for i, axis in enumerate(ax):
        axis.set_xlim(xlim)
        axis.set_ylim(ylim[0], ylim[1]*1.1)
        axis.text(0.05, 0.95, titles[i],
                   ha='left', va='top', transform=axis.transAxes, fontsize=18)
        axis.set_xlabel(xlabel, fontsize = 18)
        if i in [1]: #(1, 3):
            axis.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            axis.set_ylabel(ylabel, fontsize = 18)
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

def likePriorPost(color, absMagKinda, color_err, absMagKinda_err, apparentMagnitude, xdgmm, ndim=2, nPosteriorPoints=1000, projectedDimension=1):
    meanData, covData = testXD.matrixize(color, absMagKinda, color_err, absMagKinda_err)
    meanPrior, covPrior = testXD.matrixize(color, absMagKinda, color_err, 1e5)
    meanData = meanData[0]
    covData = covData[0]
    meanPrior = meanPrior[0]
    covPrior = covPrior[0]
    xabsMagKinda = testXD.parallax2absMagKinda(xparallaxMAS, apparentMagnitude)
    allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints)
    allMeansPrior, allAmpsPrior, allCovsPrior, summedPriorAbsMagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanPrior, covPrior, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints)
    posteriorParallax = summedPosteriorAbsmagKinda*10.**(0.2*apparentMagnitude)
    priorParallax = summedPriorAbsMagKinda*10.**(0.2*apparentMagnitude)
    likeParallax = gaussian(absMagKinda/10.**(0.2*apparentMagnitude), absMagKinda_err/10.**(0.2*apparentMagnitude), xparallaxMAS)
    return likeParallax, priorParallax, posteriorParallax

#    for label, style in zip(['paper', 'talk'],['seaborn-paper', 'seaborn-talk']):
pdf = True
style = 'seaborn-paper'
plt.style.use(style)
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['font.size'] = 25
nsubsamples = 1024
np.random.seed(0)

trueColor='darkred'
priorColor='darkgreen'
posteriorColor='royalblue'
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
if pdf:
    toy.makeplots(mtrue=mtrue, btrue=btrue, ttrue=ttrue, nexamples=nexamples,
    trueColor=trueColor, priorColor=priorColor, posteriorColor=posteriorColor, dataColor=dataColor, posteriorMapColor=posteriorMapColor)
    os.rename('toy.paper.pdf', 'paper/toy.pdf')
#----------------------------------------------


#generate raw data plot
tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()
posterior = np.load(posteriorFile)
mean = posterior['mean']
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

plot_samples(color, absMag, color_err, absMag_err, ind, contourColor='grey', rasterized=True, plot_contours=True, dataColor=dataColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, pdf=pdf)
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
fig, ax = plt.subplots()
comparePrior.dustViz(ngauss=128, quantile=0.05, iter='10th', survey='2MASS', dataFilename='All.npz', ax=ax, tgas=tgas)
fig.savefig('paper/dust.pdf', dpi=400)
fig.savefig('dust.png')
plt.close(fig)
#-------------------------------------------------------


#generate prior plot
samplex, sampley = sampleXDGMM(xdgmm, len(tgas))
titles = ["Extreme Deconvolution\n  resampling", "Extreme Deconvolution\n  cluster locations"]
plot_samples(samplex, sampley, None, None, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor=priorColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, prior=True, xdgmm=xdgmm, pdf=pdf)
if pdf: os.rename('plot_sample.pdf', 'paper/prior.pdf')
os.rename('plot_sample.png', 'prior.png')
#-------------------------------------------------------


#generate comparison prior plot
data = np.load(dustFile)
dustEBV = data['ebv']
absMagKinda, apparentMagnitude = testXD.absMagKindaArray(absmag, dustEBV, bandDictionary, tgas['parallax'])
color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)
color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
absMagKinda_err = tgas['parallax_error']*10.**(0.2*apparentMagnitude)
#-------------------------------------------------------


#M67 plot
fig, ax = plt.subplots(2,2)
ax = ax.flatten()
nPosteriorPoints = 1000
testXD.distanceTest(tgas, xdgmm, nPosteriorPoints, color, absMagKinda, color_err, absMagKinda_err, xlim_cmd, ylim_cmd, bandDictionary, absmag, mag1, mag2, plot2DPost=False, dataColor=dataColor, priorColor=priorColor, truthColor=trueColor, posteriorColor=posteriorColor, figDist=fig, axDist=ax, xlabel=xlabel_cmd, ylabel=ylabel_cmd, lw_2dlike=1)
fig.tight_layout()
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

setup_text_plots(fontsize=16, usetex=True)
plt.clf()
alpha = 0.1
alpha_points = 0.01
fig = plt.figure(figsize=(12, 5.5))
fig.subplots_adjust(left=0.1, right=0.95,
                        bottom=0.15, top=0.95,
                        wspace=0.1, hspace=0.1)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax = [ax1, ax2]
titles =  ['Exp Dec Sp Den Prior', 'CMD Prior']
for i, file in enumerate(['posteriorSimple.npz', posteriorFile]):
    data = np.load(file)
    posterior = data['posterior']
    sigma = np.sqrt(data['var'])
    mean = data['mean']
    absMag = testXD.absMagKinda2absMag(mean[positive]*10.**(0.2*apparentMagnitude))
    absMag_err = absMagError(mean[positive], sigma[positive], apparentMagnitude, absMag)
    #ax[i].scatter(color[ind], absMag[ind], c=posteriorColor, s=1, lw=0, alpha=alpha, zorder=0)
    ax[i].errorbar(color[ind], absMag[ind], xerr=color_err[ind], yerr=[absMag_err[0][ind], absMag_err[1][ind]], fmt="none", zorder=0, mew=0, ecolor=posteriorColor, alpha=0.5, elinewidth=0.5, color=posteriorColor)
    ax[i].set_xlim(xlim_cmd)
    ax[i].set_ylim(ylim_cmd[0], ylim_cmd[1]*1.1)
    ax[i].text(0.05, 0.95, titles[i], ha='left', va='top', transform=ax[i].transAxes, fontsize=18)
    ax[i].set_xlabel(xlabel_cmd, fontsize = 18)
    if i in [1]:
        ax[i].yaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax[i].set_ylabel(ylabel_cmd, fontsize = 18)
if pdf: fig.savefig('paper/comparePrior.pdf', dpi=400)
fig.savefig('comparePrior.png')
plt.close(fig)
#-------------------------------------------------------


#generate expectation plot
absMag = testXD.absMagKinda2absMag(mean[positive]*10.**(0.2*apparentMagnitude))
absMag_err = absMagError(mean[positive], sigma[positive], apparentMagnitude, absMag)
titles = ["De-noised Expectation Values", "Posterior Distributions"]
plot_samples(color, absMag, color_err, absMag_err, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor=posteriorColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, pdf=pdf)
if pdf: os.rename('plot_sample.pdf', 'paper/posteriorCMD.pdf')
os.rename('plot_sample.png', 'posterior.png')
#-------------------------------------------------------


#posterior example plot
colorBins = [0.0, 0.2, 0.4, 0.7, 1.0]
digit = np.digitize(color, colorBins)

ndim = 2
nPosteriorPoints = 1000 #number of elements in the posterior array
projectedDimension = 1  #which dimension to project the prior onto
xparallaxMAS = np.linspace(0, 10, nPosteriorPoints)

#plot likelihood and posterior in each axes
for iteration in np.arange(20, 40):
    fig, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    fig.subplots_adjust(left=0.1, right=0.9,
                                bottom=0.1, top=0.9,
                                wspace=0.4, hspace=0.5)


    plotPrior(xdgmm, ax[0], c=priorColor, lw=1)
    ax[0].set_xlim(xlim_cmd)
    ax[0].set_ylim(ylim_cmd)
    ax[0].set_xlabel(xlabel_cmd, fontsize=18)
    ax[0].set_ylabel(ylabel_cmd, fontsize=18)

    for i in range(np.max(digit)):
        currentInd = np.where((digit == i))[0]
        index = currentInd[np.random.randint(0, high=len(currentInd))]
        ax[0].scatter(color[index], absMag_dust[index], c=dataColor)
        ax[0].errorbar(color[index], absMag_dust[index], xerr=[[color_err[index], color_err[index]]], yerr=[[absMag_dust_err[0][index], absMag_dust_err[1][index]]], fmt="none", zorder=0, lw=2.0, mew=0, alpha=1.0, color=dataColor, ecolor=dataColor)
        ax[0].annotate(str(i+1), (color[index]+0.05, absMag_dust[index]+0.15), fontsize=18)
        print len(color), len(absMagKinda_dust), len(color_err), len(absMagKinda_dust_err), len(apparentMagnitude)
        likeParallax, priorParallax, posteriorParallax = likePriorPost(color[index], absMagKinda_dust[index], color_err[index], absMagKinda_dust_err[index], apparentMagnitude[index], xdgmm, ndim=2, nPosteriorPoints=1000, projectedDimension=1)

        l1, = ax[i+1].plot(xparallaxMAS, likeParallax*np.max(posteriorParallax)/np.max(likeParallax), lw=2, color=dataColor)
        l2, = ax[i+1].plot(xparallaxMAS, priorParallax*np.max(posteriorParallax)/np.max(priorParallax), lw=0.5, color=priorColor)
        l3, = ax[i+1].plot(xparallaxMAS, posteriorParallax, lw=2, color=posteriorColor)
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
        if pdf: fig.savefig('posterior_' + str(iteration) + '.pdf', dpi=400)
        fig.tight_layout()
        fig.savefig('posterior.png')
        plt.close(fig)

#-------------------------------------------------------




#delta plot
label = r'$\mathrm{ln} \, \tilde{\sigma}_{\varpi}^2 - \mathrm{ln} \, \sigma_{\varpi}^2$'

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
x = color
y = np.log(sigma**2.) - np.log(tgas['parallax_error']**2.)
levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
norm = plt.matplotlib.colors.Normalize(vmin=-1.5, vmax=1)
cmap = 'inferno'
ax[0].scatter(x, y, c=y, s=1, lw=0, alpha=0.05, norm=norm, cmap=cmap, rasterized=True)
corner.hist2d(x, y, bins=200, ax=ax[0], levels=levels, no_fill_contours=True, plot_density=False, plot_data=False, color=contourColor, rasterized=True)
ax[0].set_xlabel(xlabel_cmd, fontsize=18)
ax[0].set_ylim(-6, 2)
ax[0].set_xlim(-0.5, 2)
ax[0].set_ylabel(label, fontsize=18)
cNorm  = plt.matplotlib.colors.Normalize(vmin=0.1, vmax=2)
ax[1].scatter(x, absMag[notnans], s=1, lw=0, c=y, alpha=0.05, norm=norm, cmap=cmap, rasterized=True)
ax[1].set_xlim(xlim)
ax[1].set_ylim(ylim)
ax[1].set_xlabel(xlabel, fontsize=18)
ax[1].set_ylabel(ylabel, fontsize=18)
if pdf: fig.savefig('paper/delta.pdf', dpi=400)
fig.savefig('delta.png')
plt.close(fig)

#delta cdf plot
ratioCmd = sigma**2./tgas['parallax_error']**2.
lnratio = np.log(ratioCmd)
N = len(lnratio)
ys = np.arange(0+0.5/N, 1, 1.0/N)
sinds = np.argsort(lnratio)
f = scipy.interpolate.interp1d(lnratio[sinds], ys)
f_inv = scipy.interpolate.interp1d(ys, lnratio[sinds])
plt.plot(lnratio[sinds], ys, 'k-', lw=2)
fac2 = np.log(1/4.)
plt.plot([fac2, fac2],[-1, f(fac2)], 'k--', lw=2)
plt.plot([-6, fac2],[f(fac2), f(fac2)], 'k--' ,lw=2)
plt.plot([f_inv(0.5), f_inv(0.5)], [-1, 0.5], 'k--', lw=2)
plt.plot([-6, f_inv(0.5)], [0.5, 0.5], 'k--', lw=2)
plt.xlabel(label)
plt.xlim(-6, 2)
plt.ylim(-0.05, 1.05)
if pdf: plt.savefig('paper/deltaCDF.pdf', dpi=400)
plt.savefig('deltaCDF.png')
plt.close(fig)

#delta mean vs gaia uncertainty
y = mean - tgas['parallax']
x = tgas['parallax_error']
plt.plot(x, y, 'ko', ms=1, rasterized=True)
plt.plot([0, 1.1], [0,0])
plt.xlim(0.15, 1.05)
plt.ylim(-2.5, 2.5)
xlabel = 'Posterior Expectation Value - $\varpi_n$'
ylabel = r'$\sigma_{\varpi,n}$'
if pdf: plt.savefig('paper/deltaParallax.pdf', dpi=400)
plt.savefig('deltaParallax.png')
plt.close(fig)

#what's that feature plot
fig, ax = plt.subplots()
ax.scatter(color, absMag, s=1, lw=0, c=dataColor, alpha=alpha, zorder=0, rasterized=True)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
lowerMainSequence = (0.45, 5.5)
upperMainSequence = (-0.225, 2)
binarySequence = (0.75, 4)
redClump = (0.35, -2)
redGiantBranch = (1.0, -2)
turnOff = (0.0, 3.5)
features = [lowerMainSequence, upperMainSequence, binarySequence, redClump, redGiantBranch, turnOff]
labels = ['lower MS', 'upper MS', 'binary sequence', 'red clump', 'RGB', 'MS turn off', 'subgiant branch']
for l, f in zip(labels, features): ax.text(f[0], f[1], l, fontsize=15)
if pdf: fig.savefig('paper/whatsThatFeature.pdf', dpi=400)
fig.savefig('whatsThatFeature.png')
plt.close(fig)
