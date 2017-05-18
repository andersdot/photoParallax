import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
from dustmaps.bayestar import BayestarQuery
from astropy.coordinates import SkyCoord
import astropy.units as units
import testXD

def distanceFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'distanceQuantiles.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename

def dustFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename


ngauss = 128
thresholdSN = 0.001
survey = '2MASS'
dataFilename = 'All.npz'
survey = '2MASS'
quantile = 0.5

tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()


dustEBVnew = None
if quantile == 0.5:
    dustFile = dustFilename(ngauss, 0.05, '5th', survey, dataFilename)
    dust = np.load(dustFile)
    dustEBV = dust['ebv']
else:
    dustFile = dustFilename(ngauss, 0.05, '1st', survey, dataFilename)
    dust = np.load(dustFile)
    dustEBV = np.zeros(len(dust['ebv']))

deltaDustMax = 2

mainFig, mainAx = plt.subplots(1, 2, figsize=(15, 6))
dustDiffFig, dustDiffAx = plt.subplots(1, 2, figsize=(15, 6))
dustDistParFig = plt.figure()
dustDiffMeanFig, dustDiffMeanAx = plt.subplots(2,2, figsize=(15,6))
dustDiffMeanAx = dustDiffMeanAx.flatten()

iterNum = [1, 2, 3, 4, 5, 6, 7, 8]#, 9]
iteration = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']#, '9th']#, '10th']
color = ['black', 'grey', 'pink', 'purple', 'blue', 'cyan', 'green', 'yellow']#, 'orange']#, 'red']

b = tgas['b']*units.deg
parallax = tgas['parallax']


print np.sum(parallax < 0)
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0)
for iter, c, itN in zip(iteration, color, iterNum):

    distanceFile  = distanceFilename(ngauss, quantile, iter, survey, dataFilename)
    dustFile      = dustFilename(ngauss, quantile, iter, survey, dataFilename)
    data = np.load(dustFile)
    dustEBVnew = data['ebv']
    data = np.load(distanceFile)
    distance = data['distance']

    #distanceFile2 = 'distanceQuantiles.' + str(ngauss) + 'gauss.' + iter + '.' + survey + '.' + 'cutMatchedArrays.SN0.001.npz'
    #dustFile2     = 'dustCorrection.'    + str(ngauss) + 'gauss.' + iter + '.' + survey + '.' + 'cutMatchedArrays.SN0.001.npz'
    #data = np.load(dustFile2)
    #dustEBVnew2 = data['ebv']
    #data = np.load(distanceFile2)
    #distance2 = data['distanceQuantile']
    dustDistParFig.clf()
    dustDistParAx = dustDistParFig.add_subplot(111)
    shmag = parallax*10.**(0.2*(testXD.dustCorrect(twoMass[bandDictionary['J']['key']], dustEBVnew, 'J')))
    im = dustDistParAx.scatter(shmag, dustEBVnew - dustEBV, c=np.log10(tgas['parallax_error']), cmap='plasma_r', norm=norm, lw=0, alpha=0.5, s=5)

    #negative = parallax <= 0.03
    #negative = sn < 1.
    #dustDistParAx.scatter(np.log(distance[negative]), dustEBVnew[negative]-dustEBV[negative], c='black', lw=0, alpha=1, s=5)
    cb = plt.colorbar(im, ax=dustDistParAx)
    cb.set_label(r'log $\sigma_{\varpi}$')
    dustDistParAx.set_xlabel(r'$\varpi 10^{0.2m_{J, \mathrm{DC}}}$')
    dustDistParAx.set_ylabel(r'$\Delta$ E(B-V) [current - previous]')
    dustDistParAx.set_xlim(-100, 500)
    dustDistParAx.set_ylim(-deltaDustMax, deltaDustMax)
    dustDistParAx.axvline(x=100, lw=2, linestyle='--', zorder=0, color='black')
    plt.tight_layout()
    dustDistParFig.savefig('dustDistParallax.' + iter +'.dQ' + str(quantile) + '.png')

    dustDiffAx[0].cla()
    dustDiffAx[1].cla()
    dustDiffAx[0].scatter(distance, dustEBVnew - dustEBV, lw=0, alpha=0.5, s=1)
    dustDiffAx[1].scatter(b, dustEBVnew - dustEBV, lw=0, alpha=0.5, s=1)
    dustDiffAx[0].set_xlabel('5% Distance [kpc]')
    dustDiffAx[0].set_xlim(1e-2, 20)
    dustDiffAx[0].set_xscale('log')
    #dustDiffAx[0].set_yscale('log')
    dustDiffAx[1].set_xlabel('Galactic Latitude [deg]')
    dustDiffAx[0].set_ylabel('$\Delta$ E(B-V) [current - previous]')
    #dustDiffAx[1].set_yscale('log')
    dustDiffAx[0].set_xlim(1e-2, 20)
    dustDiffAx[0].set_ylim(-deltaDustMax, deltaDustMax)
    dustDiffAx[1].set_ylim(-deltaDustMax, deltaDustMax)
    dustDiffFig.savefig('dustDiff.' + iter + '.dQ' + str(quantile) + '.png')
    """
    dustDiffAx2[0].cla()
    dustDiffAx2[1].cla()
    dustDiffAx2[0].scatter(distance2, dustEBVnew2 - dustEBV2, lw=0, alpha=0.5, s=1, c='blue')
    dustDiffAx2[2].scatter(distance, dustEBVnew - dustEBV, lw=0, alpha=0.5, s=1, c='green')
    #dustDiffAx2[1].scatter(b, dustEBVnew2 - dustEBV2, lw=0, alpha=0.5, s=1, c='blue')
    dustDiffAx2[1].scatter(b, dustEBVnew - dustEBV, lw=0, alpha=0.5, s=1, c='green')
    dustDiffAx2[0].set_xlabel('5% Distance [kpc]')
    dustDiffAx2[2].set_xlabel('5% Distance [kpc]')
    dustDiffAx2[0].set_xlim(1e-2,)
    dustDiffAx2[2].set_xlim(1e-2,)
    dustDiffAx2[0].set_xscale('log')
    #dustDiffAx[0].set_yscale('log')
    dustDiffAx2[1].set_xlabel('Galactic Latitude [deg]')
    dustDiffAx2[0].set_ylabel('$\Delta$ E(B-V) [new - old]')
    #dustDiffAx[1].set_yscale('log')
    dustDiffAx2[0].set_xlim(1e-2,)
    dustDiffAx2[0].set_ylim(-2, 2)
    dustDiffAx2[1].set_ylim(-2, 2)
    dustDiffAx2[2].set_ylim(-2, 2)
    dustDiffFig2.savefig('dustDiffNewComp.' + iter + '.png')
    """

    dustDiffFig.savefig('dustDiff.' + iter + '.dQ' + str(quantile) + '.png')


    mainAx[0].scatter(distance, dustEBVnew - dustEBV, lw=0, alpha=0.1, s=1, color=c)
    mainAx[1].scatter(b, dustEBVnew - dustEBV, lw=0, alpha=0.1, s=1, color=c, label=iter + ' iteration')
    deltaDust = dustEBVnew - dustEBV
    dustDiffMeanAx[0].scatter(itN, np.mean(np.abs(deltaDust)))
    dustDiffMeanAx[1].scatter(itN, np.mean(np.abs(deltaDust[shmag<100])))
    dustDiffMeanAx[2].scatter(itN, np.sqrt(np.sum(deltaDust**2.)/len(deltaDust)))
    dustDiffMeanAx[3].scatter(itN, np.sqrt(np.sum(deltaDust[shmag < 100]**2.)/np.sum(shmag<100)))
    #set new dust values
    dustEBV = dustEBVnew

for i in [0,1]:
    dustDiffMeanAx[i].set_xlabel('iteration')
    dustDiffMeanAx[i].set_ylabel('$<\Delta$ E(B-V)>')
    dustDiffMeanAx[i].set_yscale('log')
for i in [2,3]:
    dustDiffMeanAx[i].set_xlabel('iteration')
    dustDiffMeanAx[i].set_ylabel('$RMS \Delta$ E(B-V)')
    dustDiffMeanAx[i].set_yscale('log')

dustDiffMeanAx[1].set_title('Giant Stars')
dustDiffMeanAx[0].set_title('All Stars')

plt.tight_layout()
dustDiffMeanFig.savefig('dustDiffMean.png')
mainAx[0].set_xlabel('5% Distance [kpc]')
mainAx[0].set_xlim(1e-2,20)
mainAx[0].set_ylim(-deltaDustMax, deltaDustMax)
mainAx[1].set_ylim(-deltaDustMax, deltaDustMax)
mainAx[0].set_xscale('log')
#mainAx[0].set_yscale('log')
mainAx[1].set_xlabel('Galactic Latitude [deg]')
mainAx[0].set_ylabel('$\Delta$ E(B-V) [current - previous]')
lgnd = mainAx[1].legend(bbox_to_anchor=(1.1, 1.05), numpoints=1, fontsize=10)
for i in range(1):
    lgnd.legendHandles[i]._sizes = [30]


mainFig.savefig('dustDiff.dQ' + str(quantile) + '.png')
