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

def plotPrior(xdgmm, ax, c='black'):
    for gg in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        ax.plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=1, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))


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
color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)[positive]
color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)[positive]
absMag = testXD.absMagKinda2absMag(tgas['parallax'][positive]*10.**(0.2*apparentMagnitude[positive]))
absMag_err = absMagError(tgas['parallax'][positive], tgas['parallax_error'][positive], apparentMagnitude[positive], absMag)
titles = ["Observed Distribution", "Obs+Noise Distribution"]

plot_samples(color, absMag, color_err, absMag_err, ind, contourColor='grey', rasterized=True, plot_contours=True, dataColor=dataColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, pdf=pdf)
if pdf: os.rename('plot_sample.pdf', 'paper/data.pdf')
os.rename('plot_sample.png', 'data.png')
#-------------------------------------------------------


#generate prior plot
xdgmm = XDGMM(filename=xdgmmFile)
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
color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)[positive]
color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)[positive]

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
    absMag = testXD.absMagKinda2absMag(mean[positive]*10.**(0.2*apparentMagnitude[positive]))
    absMag_err = absMagError(mean[positive], sigma[positive], apparentMagnitude[positive], absMag)
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
#-------------------------------------------------------


#generate expectation plot
absMag = testXD.absMagKinda2absMag(mean[positive]*10.**(0.2*apparentMagnitude[positive]))
absMag_err = absMagError(mean[positive], sigma[positive], apparentMagnitude[positive], absMag)
titles = ["De-noised Expectation Values", "Posterior Distributions"]
plot_samples(color, absMag, color_err, absMag_err, ind, contourColor='black', rasterized=True, plot_contours=True, dataColor=posteriorColor, titles=titles, xlim=xlim_cmd, ylim=ylim_cmd, xlabel=xlabel_cmd, ylabel=ylabel_cmd, pdf=pdf)
if pdf: os.rename('plot_sample.pdf', 'paper/posteriorCMD.pdf')
os.rename('plot_sample.png', 'posterior.png')
#-------------------------------------------------------


#dust plot

#posterior example plot

#delta plot

#delta cdf plot

#M67 plot

#what's that feature plot 
