from astropy.io import fits
#import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ckdtree as kdtree
import itertools
import matplotlib as mpl
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse
from extreme_deconvolution import extreme_deconvolution as ed
from matplotlib.patches import Ellipse
import pdb
from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery
from dustmaps.iphas import IPHASQuery
from dustmaps.marshall import MarshallQuery
from dustmaps.chen2014 import Chen2014Query
from astropy.coordinates import SkyCoord
import astropy.units as units
from scipy.integrate import cumtrapz
import sys
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def distMetric(sourceInd, matchedInd, apassMagnitudes, varMuMatched, p=False):
    colorChiSq = 0.0
    colorSigs = 0.0
    colors = [B_V, g_r, r_i]
    color_errors = [['e_bmag', 'e_vmag'], ['e_gmag', 'e_rmag'], ['e_rmag', 'e_imag']]
    for color, error in zip(colors, color_errors):
        colorChiSq += (color[sourceInd] - color[matchedInd])**2./(apassMagnitudes[error[0]][sourceInd]**2. + apassMagnitudes[error[1]][sourceInd]**2. +                        apassMagnitudes[error[0]][matchedInd]**2.+ apassMagnitudes[error[1]][matchedInd]**2.)
        colorSigs += np.log(apassMagnitudes[error[0]][matchedInd]**2. + apassMagnitudes[error[1]][matchedInd]**2. +                            apassMagnitudes[error[0]][sourceInd]**2. + apassMagnitudes[error[1]][sourceInd]**2.)
    absMagChiSq = (M_V[sourceInd] - M_V[matchedInd])**2./(apassMagnitudes['e_vmag'][sourceInd]**2. + apassMagnitudes['e_vmag'][matchedInd]**2. +                   varMuMatched[sourceInd] + varMuMatched[matchedInd])
    absMagSigs = np.log(apassMagnitudes['e_vmag'][sourceInd]**2. + apassMagnitudes['e_vmag'][matchedInd]**2. + varMuMatched[sourceInd] + varMuMatched[matchedInd])
    totChiSq = colorChiSq + absMagChiSq + colorSigs + absMagSigs
    if p:
        print 'the total chi2: ',totChiSq[0:5]
        print 'the color chi2: ',colorChiSq[0:5]
        print 'the color sigs: ', colorSigs[0:5]
        print 'the absmag chi2:',absMagChiSq[0:5]
        print 'the absmag sigs:',absMagSigs[0:5]

    return totChiSq

def findNeighborsAndCalculateChisqs(sourceIndex, pts, tree, apassMagnitudes, varMuMatched, nNeighbors=200, printChi=False):
    treeDistNum, treeIndex = tree.query(pts, k=nNeighbors)
    treeIndNum = treeIndex[1:]
    chisq = distMetric(sourceIndex, treeIndNum, apassMagnitudes, varMuMatched, p=printChi)
    return chisq, treeIndNum

def raveChisq(raveSourceIndex, raveTwinIndex, raveCutMatched):
    chisq = (raveCutMatched['TEFF'][raveSourceIndex] - raveCutMatched['TEFF'][raveTwinIndex])**2./(raveCutMatched['E_TEFF'][raveSourceIndex]**2. + raveCutMatched['E_TEFF'][raveTwinIndex]**2.) + (raveCutMatched['LOGG'][raveSourceIndex] - raveCutMatched['LOGG'][raveTwinIndex])**2./(raveCutMatched['E_LOGG'][raveSourceIndex]**2. + raveCutMatched['E_LOGG'][raveTwinIndex]**2.) + (raveCutMatched['FE_H'][raveSourceIndex] - raveCutMatched['FE_H'][raveTwinIndex])**2./(raveCutMatched['E_FE_H'][raveSourceIndex]**2. + raveCutMatched['E_FE_H'][raveTwinIndex]**2.) +             np.log(raveCutMatched['E_TEFF'][raveSourceIndex]**2. + raveCutMatched['E_TEFF'][raveTwinIndex]**2.) +        np.log(raveCutMatched['E_LOGG'][raveSourceIndex]**2. + raveCutMatched['E_LOGG'][raveTwinIndex]**2.) +          np.log(raveCutMatched['E_FE_H'][raveSourceIndex]**2. + raveCutMatched['E_FE_H'][raveTwinIndex]**2.)
    return chisq

def neff(weights):
    return np.sum(weights)**2./np.sum(weights**2.)

def gaussian(mean, sigma, array, amplitude=1.0):
    return amplitude/np.sqrt(2.*np.pi*sigma**2.)*np.exp(-(array - mean)**2./(2.*sigma**2.))


def pdf(mean, sigma, area, array):
    return np.sum(area/np.sqrt(2.*np.pi*sigma**2.)*np.exp(-(array - mean)**2./(2.*sigma**2.)), axis=1)

def plotPDFs(axes, raveTwinIndex, raveSourceIndex, raveCutMatched, chisqApass, npoints=1000):
    teff_prob = np.zeros(npoints)
    logg_prob = np.zeros(npoints)
    feh_prob = np.zeros(npoints)

    #guassian weight for each twin
    gaussianArea = np.exp(-chisqApass/2.)
    neff = np.sum(gaussianArea)**2./np.sum(gaussianArea**2.)

    temp = raveCutMatched['TEFF'][raveTwinIndex]
    temp_err = raveCutMatched['E_TEFF'][raveTwinIndex]
    teff_pdf = pdf(temp, temp_err, gaussianArea, teff_array[:,None])

    logg = raveCutMatched['LOGG'][raveTwinIndex]
    logg_err = raveCutMatched['E_LOGG'][raveTwinIndex]
    logg_pdf = pdf(logg, logg_err, gaussianArea, logg_array[:,None])

    feh = raveCutMatched['FE_H'][raveTwinIndex]
    feh_err = raveCutMatched['E_FE_H'][raveTwinIndex]
    feh_pdf = pdf(feh, feh_err, gaussianArea, feh_array[:,None])

    ax = axes[0]
    ax.plot(teff_array/1000., teff_pdf/np.max(teff_pdf), lw=2, label='PDF', alpha=alpha)
    ax.set_title('{:.2e}'.format(neff))
    star_gauss = gaussian(raveCutMatched['TEFF'][raveSourceIndex], raveCutMatched['E_TEFF'][raveSourceIndex], teff_array)
    ax.plot(teff_array/1000., star_gauss/np.max(star_gauss), lw=2, label='RAVE', alpha=alpha)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('T$_\mathrm{eff}$ [kK]', fontsize=20)
    ax.legend()

    ax = axes[1]
    ax.plot(logg_array, logg_pdf/np.max(logg_pdf), lw=2, label='PDF', alpha=alpha)
    star_gauss = gaussian(raveCutMatched['LOGG'][raveSourceIndex], raveCutMatched['E_LOGG'][raveSourceIndex], logg_array)
    ax.plot(logg_array, star_gauss/np.max(star_gauss), lw=2, label='RAVE', alpha=alpha)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('log g')
    ax.legend()

    ax = axes[2]
    ax.plot(feh_array, feh_pdf/np.max(feh_pdf), lw=2, label='PDF', alpha=alpha)
    star_gauss = gaussian(raveCutMatched['FE_H'][raveSourceIndex], raveCutMatched['E_FE_H'][raveSourceIndex], feh_array)
    ax.plot(feh_array, star_gauss/np.max(star_gauss), lw=2, label='RAVE', alpha=alpha)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Fe/H')
    ax.legend()

def plotRainbows(axes, apassTwinIndex, apassSourceIndex, raveTwinIndex, raveSourceIndex, apassCutMatched, raveCutMatched, varmuCutMatched, chisqApass, vmax=5):
    alpha_points = 0.3
    alpha_bars = 0.25
    vmax = 5

    b_v_lim = [0.25, 1.5]
    g_r_lim = None #[0, 1.5]

    r_i_lim = None #[-0.25, 0.75]
    M_v_lim = None #[10, 2]

    teff_lim = [7, 4] #kK
    log_g_lim = [6, 3]
    #plot B-V vs g-r for the source and twins
    """
    ax = axes[0]
    ax.set_title('Tycho2 ID: ' + tgasMatched['tycho2_id'][apassSourceIndex])
    ax.scatter(g_r[apassTwinIndex], r_i[apassTwinIndex], c=chisqApass - np.min(chisqApass), cmap='plasma',
               norm=mpl.colors.Normalize(vmax=vmax), lw=0, zorder=100, alpha=alpha_points)
    ax.errorbar(g_r[apassTwinIndex], r_i[apassTwinIndex],
                xerr = np.sqrt(apassCutMatched['e_gmag'][apassTwinIndex]**2. + apassCutMatched['e_rmag'][apassTwinIndex]**2.),
                yerr = np.sqrt(apassCutMatched['e_rmag'][apassTwinIndex]**2. + apassCutMatched['e_imag'][apassTwinIndex]**2.),
                fmt="none", ecolor='black', zorder=0, lw=0.5, mew=0, alpha=alpha_bars)
    ax.errorbar(g_r[star], r_i[star],
                xerr = np.sqrt(apassCutMatched['e_bmag'][apassSourceIndex]**2. + apassCutMatched['e_vmag'][apassSourceIndex]**2.),
                yerr = np.sqrt(apassCutMatched['e_gmag'][apassSourceIndex]**2. + apassCutMatched['e_rmag'][apassSourceIndex]**2.), fmt='o', color='black',lw=4)

    ax.scatter(g_r[apassSourceIndex], r_i[apassSourceIndex], s=400, c='black')
    ax.set_xlabel('g - r')
    ax.set_ylabel('r - i')
    ax.set_xlim(g_r_lim)
    ax.set_ylim(r_i_lim)
    ax.grid()
    """
    #plot B-V and M_v for the source and twins
    ax = axes[0]
    ax.scatter(B_V[apassTwinIndex], M_V[apassTwinIndex], c=chisqApass - np.min(chisqApass), cmap='plasma',
               norm=mpl.colors.Normalize(vmax=vmax), lw=0, zorder=100, alpha=alpha_points)
    ax.errorbar(B_V[apassTwinIndex], M_V[apassTwinIndex],
                xerr = np.sqrt(apassCutMatched['e_bmag'][apassTwinIndex]**2. + apassCutMatched['e_vmag'][apassTwinIndex]**2.),
                yerr = np.sqrt(apassCutMatched['e_vmag'][apassTwinIndex]**2. + varmuCutMatched[apassTwinIndex]),
                fmt="none", ecolor='black', zorder=0, lw=0.5, mew=0, alpha=alpha_bars)
    ax.errorbar(B_V[apassSourceIndex], M_V[apassSourceIndex],
                xerr = np.sqrt(apassCutMatched['e_bmag'][apassSourceIndex]**2. + apassCutMatched['e_vmag'][apassSourceIndex]**2.),
                yerr = np.sqrt(apassCutMatched['e_vmag'][apassSourceIndex]**2. + varmuCutMatched[apassSourceIndex]),
                fmt='o', color='black', lw=4)
    ax.scatter(B_V[apassSourceIndex], M_V[apassSourceIndex], s=400, c='black')
    ax.set_xlabel('B - V')
    ax.set_ylabel('M_V')
    ax.invert_yaxis()
    ax.set_xlim(b_v_lim)
    ax.set_ylim(M_v_lim)
    ax.grid()

    #plot Teff vs log g for the source and twins
    raveTeff = raveCutMatched['TEFF'][raveTwinIndex]
    raveLogG = raveCutMatched['LOGG'][raveTwinIndex]

    ax = axes[1]
    ax.scatter(raveTeff/1000., raveLogG, c=chisqApass - np.min(chisqApass), cmap='plasma',
               norm=mpl.colors.Normalize(vmax=vmax), lw=0, zorder=100, alpha=alpha_points)
    ax.errorbar(raveTeff/1000., raveLogG,
                xerr=raveCutMatched['E_TEFF'][raveTwinIndex]/1000., yerr=raveCutMatched['E_LOGG'][raveTwinIndex],
                fmt='none', ecolor='black', zorder=0, lw=0.5, mew=0, alpha=alpha_bars)
    ax.scatter(raveCutMatched['TEFF'][raveSourceIndex]/1000., raveCutMatched['LOGG'][raveSourceIndex], s=400, c='black')
    ax.errorbar(raveCutMatched['TEFF'][raveSourceIndex]/1000., raveCutMatched['LOGG'][raveSourceIndex],
                xerr=raveCutMatched['E_TEFF'][raveSourceIndex]/1000., yerr=raveCutMatched['E_LOGG'][raveSourceIndex],
               fmt='o', color='black', lw=4)
    ax.set_ylabel('log g', fontsize=15)
    ax.set_xlabel('Teff [kK]', fontsize=15)
    ax.set_xlim(teff_lim)
    ax.set_ylim(log_g_lim)
    ax.grid()

    #plot B-V vs Fe/H for the source and twins
    ax = axes[2]
    ax.errorbar(B_V[apassTwinIndex], raveCutMatched['FE_H'][raveTwinIndex], yerr=raveCutMatched['E_FE_H'][raveTwinIndex],
              xerr = np.sqrt(apassCutMatched['e_bmag'][apassTwinIndex]**2. + apassCutMatched['e_vmag'][apassTwinIndex]**2.),
              fmt='none', ecolor='black', zorder=0, lw=0.5, mew=0, alpha=alpha_bars)
    ax.scatter(B_V[apassTwinIndex], raveCutMatched['FE_H'][raveTwinIndex], c=chisqApass - np.min(chisqApass), cmap='plasma',
               norm=mpl.colors.Normalize(vmax=vmax), lw=0, zorder=100, alpha=alpha_points)
    ax.scatter(B_V[apassSourceIndex], raveCutMatched['FE_H'][raveSourceIndex], s=400, c='black')
    ax.errorbar(B_V[apassSourceIndex], raveCutMatched['FE_H'][raveSourceIndex], yerr=raveCutMatched['E_FE_H'][raveSourceIndex],
                xerr = np.sqrt(apassCutMatched['e_bmag'][apassSourceIndex]**2. + apassCutMatched['e_vmag'][apassSourceIndex]**2.),
                fmt='o', color='black', lw=4)
    ax.set_xlabel('B - V', fontsize=15)
    ax.set_ylabel('[Fe/H]', fontsize=15)
    ax.grid()

def draw_ellipse(mu, C, scales=[1, 2, 3], ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    # find principal components and rotation angle of ellipse
    sigma_x2 = C[0, 0]
    sigma_y2 = C[1, 1]
    sigma_xy = C[0, 1]
    #print sigma_x2, sigma_y2, sigma_xy
    alpha = 0.5 * np.arctan2(2 * sigma_xy,
                             (sigma_x2 - sigma_y2))
    tmp1 = 0.5 * (sigma_x2 + sigma_y2)
    tmp2 = np.sqrt(0.25 * (sigma_x2 - sigma_y2) ** 2 + sigma_xy ** 2)
    #print tmp1, tmp2
    sigma1 = np.sqrt(np.abs(tmp1 + tmp2))
    sigma2 = np.sqrt(np.abs(tmp1 - tmp2))
    #print sigma1, sigma2
    for scale in scales:
        ax.add_patch(Ellipse((mu[0], mu[1]),
                             2 * scale * sigma1, 2 * scale * sigma2,
                             alpha * 180. / np.pi,
                             **kwargs))

def XD(raveTwins, raveCutMatched, chisqApass, ngauss=2):

    amp_guess = np.zeros(ngauss)[:,None] + 1.
    mean_guess = np.array([4.5, 4.0, 0.0])[:,None] #np.array([[4.5, 4.0, 0], [4.5, 3.5, 0]])

    temp = raveCutMatched['TEFF'][raveTwins]/1000.
    temp_err = raveCutMatched['E_TEFF'][raveTwins]/1000.

    logg = raveCutMatched['LOGG'][raveTwins]
    logg_err = raveCutMatched['E_LOGG'][raveTwins]

    feh = raveCutMatched['FE_H'][raveTwins]
    feh_err = raveCutMatched['E_FE_H'][raveTwins]

    gaussianArea = np.exp(-chisqApass/2.)

    X = np.vstack([temp, logg, feh]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:,diag,diag] = np.vstack([temp_err**2., logg_err**2., feh_err**2.]).T

    cov_guess = np.zeros(((ngauss,) + X.shape[-1:] + X.shape[-1:]))
    cov_guess[:,diag,diag] = 1.0

    ed(X, Xerr, amp_guess, mean_guess, cov_guess, weight=gaussianArea)

    return amp_guess, mean_guess, cov_guess

def plotXD2d(ax, raveTwins, raveStar, raveCutMatched, chisqApass, mean, cov, ngauss=2, fracMaxPlot=0.01, size=50, alpha_points=1., alpha_ellipse=1.):

    temp = raveCutMatched['TEFF'][raveTwins]/1000.
    temp_err = raveCutMatched['E_TEFF'][raveTwins]/1000.

    logg = raveCutMatched['LOGG'][raveTwins]
    logg_err = raveCutMatched['E_LOGG'][raveTwins]

    feh = raveCutMatched['FE_H'][raveTwins]
    feh_err = raveCutMatched['E_FE_H'][raveTwins]

    gaussianArea = np.exp(-chisqApass/2.)

    weighty = gaussianArea > fracMaxPlot*np.max(gaussianArea)

    X = np.vstack([temp, logg, feh]).T
    ax[0].set_title(neff(gaussianArea))
    ax[0].scatter(X[:,0][weighty], X[:,1][weighty], cmap='plasma_r', c = gaussianArea[weighty],
               norm=mpl.colors.Normalize(), lw=0, alpha=alpha_points,s=size)
    ax[0].errorbar(raveCutMatched['TEFF'][raveStar]/1000., raveCutMatched['LOGG'][raveStar],
                  xerr=raveCutMatched['E_TEFF'][raveStar]/1000., yerr=raveCutMatched['E_LOGG'][raveStar],fmt='o',lw=2, zorder=100)
    for scale, alpha in zip([1, 2], [1.0, 0.5]):
        for i in range(ngauss):
            draw_ellipse(mean[i][0:2], cov[i][0:2], scales=[scale], ax=ax[0],
                 ec='k', fc="None", alpha=alpha_ellipse, zorder=99, lw=2)

    ax[1].scatter(X[:,0][weighty], X[:,2][weighty], cmap='plasma_r', c = gaussianArea[weighty],
               norm=mpl.colors.Normalize(), lw=0, alpha=alpha_points, s=size)
    ax[1].errorbar(raveCutMatched['TEFF'][raveStar]/1000., raveCutMatched['FE_H'][raveStar],
                  xerr=raveCutMatched['E_TEFF'][raveStar]/1000., yerr=raveCutMatched['E_FE_H'][raveStar],fmt='o',lw=2, zorder=100)

    for scale, alpha in zip([1, 2], [1.0, 0.5]):
        for i in range(ngauss):
            draw_ellipse(mean[i][[0,2]], cov[i][[0,2]], scales=[scale], ax=ax[1],
                 ec='k', fc="None", alpha=alpha_ellipse, zorder=99, lw=2)

    ax[2].scatter(X[:,1][weighty], X[:,2][weighty], cmap='plasma_r', c = gaussianArea[weighty],
               norm=mpl.colors.Normalize(), lw=0, alpha=alpha_points, s=size)
    ax[2].errorbar(raveCutMatched['LOGG'][raveStar], raveCutMatched['FE_H'][raveStar],
                  xerr=raveCutMatched['E_LOGG'][raveStar], yerr=raveCutMatched['E_FE_H'][raveStar],fmt='o',lw=2, zorder=100)

    for scale, alpha in zip([1, 2], [1.0, 0.5]):
        for i in range(ngauss):
            draw_ellipse(mean[i][1:3], cov[i][1:3], scales=[scale], ax=ax[2],
                 ec='k', fc="None", alpha=alpha_ellipse, zorder=99, lw=2)

    ax[0].set_xlim(teff_lim)
    ax[0].set_xlabel('T$_\mathrm{eff}$')
    ax[1].set_xlim(teff_lim)
    ax[1].set_xlabel('T$_\mathrm{eff}$')
    ax[2].set_xlim(log_g_lim)
    ax[2].set_xlabel('log g')
    ax[0].set_ylim(log_g_lim)
    ax[0].set_ylabel('log g')
    ax[1].set_ylim(feh_lim)
    ax[1].set_ylabel('Fe/H')
    ax[2].set_ylim(feh_lim)
    ax[2].set_ylabel('Fe/H')
    plt.tight_layout()

def tgasDistance(ndist=1024):
    #read in Adrian's distances from sampling the posterior
    nfiles = 16
    dist = None #np.zeros(len(tgasMatched), ndist)
    for j in range(1,nfiles+1):
        with h5py.File("distance-samples-{:02d}.hdf5".format(j)) as f:
            if dist is None:
                dist = f['distance'][:,:ndist]
            else:
                dist = np.concatenate((dist, f['distance'][:,:ndist]))
    return dist

def observationsCutMatched(SNthreshold=1., filename='cutMatchedArrays.npz'):

    #read in TGAS data for and matched sample for magnitudes
    tgas = fits.getdata("stacked_tgas.fits", 1)
    tgasRave = fits.getdata('tgas-rave.fits', 1)
    tgasApass = fits.getdata('tgas-matched-apass-dr9.fits')
    tgasWise = fits.getdata('tgas-matched-wise.fits')
    tgas2mass = fits.getdata('tgas-matched-2mass.fits')
    distances = tgasDistance(ndist=1024)
    medianDist = np.median(distances, axis=1)
    #cut out low logg and temperatures outside well populated area
    nonNans = ~np.isnan(tgasRave['TEFF']) & ~np.isnan(tgasRave['LOGG']) & ~np.isnan(tgasRave['FE_H'])
    #dwarfs = (tgasRave['LOGG'] < maxlogg) & (tgasRave['LOGG'] > minlogg) & (tgasRave['TEFF'] > mintemp)
    tgasRave = tgasRave[nonNans] # & dwarfs]

    #various cuts to select sample
    magSN = SNthreshold
    sigMax = 1.086/magSN
    maxDist = 8000. #1000. #pc
    fracErrorDistance = 1.
    minDist = 0.0 #pc
    parallaxSN = SNthreshold
    galacticLatMin = 0. #degrees
    galacticLatMax = None #degrees

    #current cut: no magnitudes are NaNs, 0 < errors < sigMax,
    noNans = ~np.isnan(tgasApass['bmag']) & ~np.isnan(tgasApass['vmag']) & ~np.isnan(tgasApass['gmag']) & ~np.isnan(tgasApass['rmag']) & ~np.isnan(tgasApass['imag'])

    posErrors = (tgasApass['e_bmag'] > 0) & (tgasApass['e_vmag'] > 0) & (tgasApass['e_gmag'] > 0) & (tgasApass['e_rmag'] > 0) & (tgasApass['e_imag'] > 0)

    lowPhotError = (tgasApass['e_bmag'] < sigMax) & (tgasApass['e_vmag'] < sigMax) & (tgasApass['e_gmag'] < sigMax) & (tgasApass['e_rmag'] < sigMax) & (tgasApass['e_imag'] < sigMax)

    #lowPhotError_IR = (tgas2mass['j_cmsig'] < sigMax) & (tgas2mass['h_cmsig'] < sigMax) & (tgas2mass['k_cmsig'] < sigMax)  & (tgasWise['w1sigmpro'] < sigMax) & (tgasWise['w2sigmpro'] < sigMax) & (tgasWise['w3sigmpro'] < sigMax)

    lowPhotError_IR = (tgas2mass['j_cmsig'] < sigMax) & (tgas2mass['k_cmsig'] < sigMax)

    noDust = (medianDist < maxDist) & (np.abs(tgas['b']) > galacticLatMin)
    apassMatch = tgasApass['matched']
    wiseMatch = tgasWise['matched']
    twoMassMatch = tgas2mass['matched']
    parallaxErr = np.sqrt(tgas['parallax_error']**2. + 0.3**2.)
    goodDistance = tgas['parallax']/parallaxErr > parallaxSN
    hasDust = medianDist >= minDist
    raveMatch = np.in1d(tgas['source_id'], tgasRave['source_id'])
    if galacticLatMax: inDisk = np.abs(tgas['b'] <= galacticLatMax)
    matched = goodDistance & lowPhotError_IR #& apassMatch


    tgasMatched = tgas[matched]
    magsMatched = tgasApass[matched]
    wiseMatched = tgasWise[matched]
    twoMassMatched = tgas2mass[matched]
    distMatched = distances[matched]
    raveInd = np.in1d(tgasRave['source_id'], tgasMatched['source_id'])
    raveMatched = tgasRave[raveInd]

    print 'Number of tgas stars: ', len(tgas)
    print 'Number of matched stars: ', np.sum(matched)
    print 'Percent matched = ', 100 - (len(tgas) - np.sum(matched))/np.float(len(tgas))*100., '%'

    np.savez(filename, tgasCutMatched=tgasMatched, apassCutMatched=magsMatched, raveCutMatched=raveMatched, twoMassCutMatched=twoMassMatched, wiseCutMatched=wiseMatched, distCutMatched=distMatched)
    return tgasMatched, magsMatched, raveMatched, twoMassMatched, wiseMatched, distMatched

def crossMatchCheck(apassCutMatched, twoMassCutMatched, wiseCutMatched):
    #plot broad colors to check that cross matching was done properly
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Check Matching to IR surveys', fontsize=20, y=1.08)
    ax[0].scatter(apassCutMatched['bmag'] - apassCutMatched['vmag'], twoMassCutMatched['j_mag'] - twoMassCutMatched['k_mag'], alpha=0.25, lw=0)
    ax[0].set_xlabel('B - V')
    ax[0].set_ylabel('J - K')

    ax[1].scatter(apassCutMatched['bmag'] - apassCutMatched['vmag'], apassCutMatched['bmag'] - twoMassCutMatched['k_mag'], alpha=0.25, lw=0)
    ax[1].set_xlabel('B - V')
    ax[1].set_ylabel('B - K')

    ax[2].scatter(apassCutMatched['bmag'] - apassCutMatched['vmag'], apassCutMatched['bmag'] - wiseCutMatched['w3mpro'], alpha=0.25, lw=0)
    ax[2].set_xlabel('B - V')
    ax[2].set_ylabel('B - W3')


    plt.tight_layout()
    plt.savefig('IRmatchCheck.png')

def distanceModulus(distCutMatched):
    medianDistMatched = np.median(distCutMatched, axis=1)
    muMatched = 5. * np.log10(distCutMatched / 10.) # 10 pc is mu = 0
    meanMuMatched = np.mean(muMatched, axis=1)
    varMuMatched = np.mean((muMatched - meanMuMatched[:,None]) ** 2, axis=1)
    return meanMuMatched, varMuMatched

def dust(l, b, distance, plot=False, max_samples=2, mode='median', model='bayes'):
    if model == 'sfd':
        c = SkyCoord(l, b,
                frame='galactic')
        sfd = SFDQuery()
        dust = sfd(c)

    if model == 'bayes':
        c = SkyCoord(l, b,
                distance = distance,
                frame='galactic')
        bayes = BayestarQuery(max_samples=max_samples)
        dust = bayes(c, mode=mode)
    if model == 'iphas':
        c = SkyCoord(l, b,
                distance = distance,
                frame='galactic')
        iphas = IPHASQuery()
        dust = iphas(c, mode=mode)

    if model == 'marshall':
        c = SkyCoord(l, b,
                distance = distance,
                frame='galactic')
        marshall = MarshallQuery()
        dust = marshall(c)

    if model == 'chen':
        c = SkyCoord(l, b,
                distance = distance,
                frame='galactic')
        chen = Chen2014Query()
        dust = chen(c)
    #cNoDist = SkyCoord(l, b,
    #        frame='galactic')
    #bayesDustNoDist = bayes(cNoDist, mode=mode)

    #!!!!! Do something else than setting it equal to 0 !!!!!
    #if len(bayesDust) > 1: bayesDust[np.isnan(bayesDust)] = 0.0

    if plot:
        fig, ax = plt.subplots(3, figsize=(5, 7.5))

        ax[0].hist(np.log10(sfd(c)), bins=100, log=True, histtype='step')
        ax[1].hist(np.log10(bayesDustNoDist[bayesDustNoDist>0]), bins=100, log=True, histtype='step')
        ax[2].hist(np.log10(bayesDust[bayesDust >0]), bins=100, log=True, histtype='step')

        ax[0].set_xlabel('SFD Dust Attenuation')
        ax[1].set_xlabel('Bayestar Dust Attenuation No Distance')
        ax[2].set_xlabel('Bayestar Dust Attenuation')
        for a in ax: a.set_xlim(-4, 0.0)
        hist, bins = np.histogram(magsMatched['bmag'], bins=100)
        plt.hist(magsMatched['bmag'], bins=bins, histtype='step')
        plt.hist(magsMatched['bmag'] - B_RedCoeff*bayesDust, bins=bins, histtype='step')
        plt.tight_layout()
    return dust

def dustTightenMS(B_V_dust, M_V_dust, B_V, M_V):
    #check dust tightens main sequence
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Check Dust Tightens MS', fontsize=20, y=1.08)
    ax[0].scatter(B_V_dust, M_V_dust, alpha=0.1, lw=0)
    ax[0].set_xlabel('B - V')
    ax[0].set_ylabel('M_V')
    ax[0].set_title('Dust Corrected')
    ax[0].invert_yaxis()

    ax[1].scatter(B_V, M_V, alpha=0.1, lw=0)
    ax[1].set_xlabel('B - V')
    ax[1].set_ylabel('M_V')
    ax[1].set_title('No Dust Correction')
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig('dustCorrected.png')

def mainSequence(B_V, M_V, raveCutMatched):
    alpha = 0.1
    fig, axes = plt.subplots(3,2, figsize=(12.5, 17))
    axes = axes.flatten()
    axes[0].scatter(B_V,M_V, alpha=alpha, lw=0)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('B-V')
    axes[0].set_ylabel('M$_\mathrm{V}$')
    axes[0].set_title('Apass HR Diagram')

    axes[1].scatter(raveCutMatched['TEFF']/1000., raveCutMatched['LOGG'], alpha=alpha, lw=0)
    axes[1].invert_yaxis()
    axes[1].invert_xaxis()
    axes[1].set_xlabel('T$_\mathrm{eff}$ [kK]')
    axes[1].set_ylabel('log g')
    axes[1].set_title('Rave HR Diagram')

    axes[2].scatter(B_V, raveCutMatched['TEFF']/1000., alpha=alpha, lw=0)
    #axes[2].invert_xaxis()
    axes[2].set_ylabel('T$_\mathrm{eff}$ [kK]')
    axes[2].set_xlabel('B - V')
    axes[2].set_title('Apass Temp vs Rave Temp')

    axes[3].scatter(M_V, raveCutMatched['LOGG'], alpha=alpha, lw=0)
    axes[3].set_xlabel('M$_\mathrm{V}$')
    axes[3].set_ylabel('log g')
    axes[3].invert_yaxis()
    axes[3].invert_xaxis()
    axes[3].set_title('Apass log g vs Rave log g')

    axes[4].scatter(M_V, raveCutMatched['FE_H'], alpha=alpha, lw=0)
    axes[4].invert_xaxis()
    axes[4].set_xlabel('M$_\mathrm{V}$')
    axes[4].set_ylabel('Fe/H')


    axes[5].scatter(B_V, raveCutMatched['FE_H'], alpha=alpha, lw=0)
    #axes[5].invert_xaxis()
    axes[5].set_xlabel('B - V')
    axes[5].set_ylabel('Fe/H')
    plt.tight_layout()
    plt.savefig('AllStarsMatched.png')

def apassSourceTwinIndex(M_V, B_V, g_r, r_i, apassCutMatched, varmuCutMatched, nstars=100, nydim=100, nNeighbors=1024, printChi=False):
    apassSourceIndex = np.zeros(nstars, dtype='int32')
    apassTwinIndex = np.zeros((nstars, nydim-1), dtype='int32')
    chisqApass = np.zeros((nstars, nydim-1))

    for star in np.arange(nstars):
        pts = [M_V[star], B_V[star], g_r[star], r_i[star]]
        chisq, treeIndex = findNeighborsAndCalculateChisqs(star, pts, treeColor, apassCutMatched, varmuCutMatched, nNeighbors=nNeighbors, printChi=False)
        sort = np.argsort(treeIndex)
        chisqApass[star,:] = chisq[sort] #don't include self, zero is closest
        #treeIndex = treeIndex[chisq < chisqThreshold]
        apassTwinIndex[star,:] = treeIndex[sort]
        apassSourceIndex[star] = star
    return apassSourceIndex, apassTwinIndex, chisqApass

def raveSourceTwinIndex(apassSourceIndex, apassTwinIndex, raveCutMatched, tgasCutMatched, nstars=100, nydim=100):
    raveSourceIndex = np.zeros(nstars, dtype='int32')
    raveTwinIndex = np.zeros((nstars, nydim-1), dtype='int32')
    chisqRave = np.zeros((nstars, nydim-1))
    for i, (s, m) in enumerate(zip(apassSourceIndex, apassTwinIndex)):
        raveSourceIndex[i] = np.where(np.in1d(raveCutMatched['source_id'], tgasCutMatched[s]['source_id']))[0]
        raveTwinIndex[i, :] = np.where(np.in1d(raveCutMatched['source_id'], tgasCutMatched[m]['source_id']))[0]
        if np.sum(raveCutMatched['source_id'][raveTwinIndex[i,:]] - tgasCutMatched[m]['source_id']) != 0: print 'Rave not sorted like Apass'
        chisqRave[i, :] = raveChisq(raveSourceIndex[i], raveTwinIndex[i,:], raveCutMatched)
    return raveSourceIndex, raveTwinIndex, chisqRave

def plotChisqApassVsRave(chisqApass, chisqRave):
    fig, ax = plt.subplots()
    for chiA, chiR in zip(chisqApass, chisqRave):
        ax.scatter(chiA, chiR, alpha=0.01, s=1)
    #ax.set_xlim(0, 10)
    #ax.set_ylim(0, 15)
    ax.set_xlabel('Chisq Distance Apass')
    ax.set_ylabel('Chisq Distance Rave')
    plt.tight_layout()
    plt.savefig('chisqApassVsRave.png')

def plotComparisons(indices, apassTwinIndex, apassSourceIndex, raveTwinIndex, raveSourceIndex, apassCutMatched, raveCutMatched, varmuCutMatched, chisqApass, nplot=10, filename='plot.png', ngauss=2):
    fig, axes = plt.subplots(nplot, 6, figsize=(30, nplot*5))
    for plotNumber, j in enumerate(indices):
        try:
            plotRainbows(axes[plotNumber][0:3], apassTwinIndex[j], apassSourceIndex[j], raveTwinIndex[j], raveSourceIndex[j], apassCutMatched, raveCutMatched, varmuCutMatched, chisqApass[j], vmax=10)
            amp, mean, cov = XD(raveTwinIndex[j], raveCutMatched, chisqApass[j], ngauss=ngauss)
            plotXD2d(axes[plotNumber][3:6], raveTwinIndex[j], raveSourceIndex[j], raveCutMatched, chisqApass[j], mean, cov, ngauss=ngauss, fracMaxPlot=0.01)
        except ValueError:
            pdb.set_trace()
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    try: plot = np.bool(sys.argv[1])
    except IndexError: plot = False
    b_v_lim = [0.25, 1.5]
    g_r_lim = None #[0, 1.5]

    r_i_lim = None #[-0.25, 0.75]
    M_v_lim = None #[10, 2]

    teff_lim = [7, 4] #kKd
    log_g_lim = [6, 3]
    feh_lim = [-1.5, 1]

    maxlogg = 20
    minlogg = 1
    mintemp = 100
    SNthreshold = 4
    filename = 'cutMatchedArrays.' + str(minlogg) + '_' + str(maxlogg) + '_' + str(mintemp) + '_' + str(SNthreshold) + '.npz'

    try:
        cutMatchedArrays = np.load(filename)
        tgasCutMatched = cutMatchedArrays['tgasCutMatched']
        apassCutMatched = cutMatchedArrays['apassCutMatched']
        raveCutMatched = cutMatchedArrays['raveCutMatched']
        twoMassCutMatched = cutMatchedArrays['twoMassCutMatched']
        wiseCutMatched = cutMatchedArrays['wiseCutMatched']
        distCutMatched = cutMatchedArrays['distCutMatched']
    except IOError:
        tgasCutMatched, apassCutMatched, raveCutMatched, twoMassCutMatched, wiseCutMatched, distCutMatched = observationsCutMatched(maxlogg=maxlogg, minlogg=minlogg, mintemp=mintemp, SNthreshold=SNthreshold, filename=filename)
    print 'Number of Matched stars is: ', len(tgasCutMatched)

    #plot broad colors to check that cross matching was done properly
    if plot: crossMatchCheck(apassCutMatched, twoMassCutMatched, wiseCutMatched)
    meanMuMatched, varmuCutMatched = distanceModulus(distCutMatched)

    apassMagKeys = ['bmag', 'gmag', 'vmag', 'rmag', 'imag']
    apassErrorKeys = ['e_bmag', 'e_gmag', 'e_vmag', 'e_rmag', 'e_imag']
    wavelength = [420., 475., 520., 658., 806.]

    #include dust
    #Assuming an R_V of 3.1, good assumption for Milky Way so say Schlafly+Finkbeiner
    B_RedCoeff = 3.626
    V_RedCoeff = 2.742
    g_RedCoeff = 3.303
    r_RedCoeff = 2.285
    i_RedCoeff = 1.698
    bayesDust = dust(tgasCutMatched['l']*units.deg, tgasCutMatched['b']*units.deg, np.median(distCutMatched, axis=1)*units.pc)
    M_V = apassCutMatched['vmag'] - V_RedCoeff*bayesDust - meanMuMatched
    B_V = apassCutMatched['bmag'] - B_RedCoeff*bayesDust - (apassCutMatched['vmag'] - V_RedCoeff*bayesDust)
    g_r = apassCutMatched['gmag'] - g_RedCoeff*bayesDust - (apassCutMatched['rmag'] - r_RedCoeff*bayesDust)
    r_i = apassCutMatched['rmag'] - r_RedCoeff*bayesDust - (apassCutMatched['imag'] - i_RedCoeff*bayesDust)


    #plot mainsequence for all matched stars in both rave and apass
    if plot: mainSequence(B_V, M_V, raveCutMatched)
    #check dust tightens main sequence
    if plot: dustTightenMS(B_V, M_V, apassCutMatched['bmag'] - apassCutMatched['vmag'], apassCutMatched['vmag'] - meanMuMatched)

    #build tree on apass photometry + GAIA
    treeColor = kdtree.cKDTree(data=zip(M_V, B_V, g_r, r_i))
    treeRave = kdtree.cKDTree(data=zip(M_V, B_V, g_r, r_i, raveCutMatched['LOGG'], raveCutMatched['TEFF']))

    #number of stars to find twins for
    nstars = len(apassCutMatched)

    #number of twins to grab for each source star
    nNeighbors = 1024
    if nNeighbors > nstars: nNeighbors = nstars
    nydim = nNeighbors

    #chisqThreshold = 100
    apassSourceIndex, apassTwinIndex, chisqApass = apassSourceTwinIndex(M_V, B_V, g_r, r_i, apassCutMatched, varmuCutMatched, nstars=nstars, nydim=nydim, nNeighbors=nNeighbors)
    raveSourceIndex, raveTwinIndex, chisqRave = raveSourceTwinIndex(apassSourceIndex, apassTwinIndex, raveCutMatched, tgasCutMatched, nstars=nstars, nydim=nydim)
    nx = 10000
    nmodel = 20
    x_model = np.linspace(1, 10, nx)

    mu = np.linspace(3, 6, nmodel)
    sigma = np.linspace(0.01, 0.5, nmodel)
    posterior = np.zeros((nmodel,nmodel))
    for obsIndex in [6]:
        x_obs = raveCutMatched[raveTwinIndex[obsIndex]]['TEFF']/1000.
        sigma_obs = raveCutMatched[raveTwinIndex[obsIndex]]['E_TEFF']/1000.
        x_rave = raveCutMatched[raveSourceIndex[obsIndex]]['TEFF']/1000.
        sigma_rave = raveCutMatched[raveSourceIndex[obsIndex]]['E_TEFF']/1000.
        plot=False
        weight_obs = np.exp(-0.5*chisqApass[obsIndex])
        for m, mean in enumerate(mu):
            for s, sig in enumerate(sigma):

                #integrand = gaussian(x_obs[:, None], sigma_obs[:, None], x_model, amplitude=weight_obs[:, None])*gaussian(mean, variance, x_model)
                #integral = cumtrapz(integrand, x=x_model)
                #loglikelihood = np.log(integral[:,-1])
                loglikelihood = np.log(gaussian(mean, sig + sigma_obs, x_obs, amplitude=weight_obs))
                posterior[m, s] = np.sum(loglikelihood)
                #if plot:
                #    fig, ax = plt.subplots(2)
                #    for foo in np.arange(nNeighbors-1): ax[0].plot(x_model, integrand[foo], alpha=0.5, color='blue')
                #    ax[1].scatter(loglikelihood, np.log(weight_obs))
                #    plt.show()
        fig, ax = plt.subplots(2)
        maxIndex = np.where(posterior == np.max(posterior))
        ax[0].plot(x_model, gaussian(mu[maxIndex[0]], sigma[maxIndex[1]], x_model, amplitude=1), color='black', lw=2, label='XD')
        for foo in range(len(x_obs)): ax[1].plot(x_model, gaussian(x_obs[foo], sigma_obs[foo], x_model, amplitude = weight_obs[foo]), alpha=0.1, color='blue')
        ax[0].plot(x_model, gaussian(x_rave, sigma_rave, x_model, amplitude=1), color='blue', linestyle='--', lw=2, label='Rave')
        ax[1].set_xlabel('TEFF [kK]')
        ax[1].set_yscale('log')
        ax[1].set_ylim(1,)
        # plt.tight_layout()
        #print x_obs[np.argsort(weight_obs)[::-1]][0:100], sigma_obs[np.argsort(weight_obs)[::-1]][0:100]
        plt.show()
    pdb.set_trace()
    neffRave = np.zeros(nstars)
    neffApass = np.zeros(nstars)
    for i in range(nstars):
        neffRave[i] = neff(np.exp(-0.5*chisqRave[i]))
        neffApass[i] = neff(np.exp(-0.5*chisqApass[i]))
    maxNeffRave = 50.
    minNeffRave = 150.

    maxNeffApass = 25.
    minNeffApass = 100.

    if plot:
        logg = raveCutMatched['LOGG'][raveSourceIndex]
        varpi = tgasCutMatched['parallax'][apassSourceIndex]
        teff = raveCutMatched['TEFF'][raveSourceIndex]
        dwarfs = (logg < 5.) & (logg > 4.2) & (teff > 4500)
        warm = teff > 4500

        fig, ax = plt.subplots(figsize=(7, 5))
        points = ax.scatter(neffApass, neffRave, lw=0, alpha=0.5, c=logg, norm=mpl.colors.Normalize(), cmap='cool')
        ax.fill_between([np.min(neffApass), maxNeffApass], [np.max(neffRave), np.max(neffRave)], y2 = [minNeffRave, minNeffRave], color='black', alpha=0.1)
        ax.fill_between([minNeffApass, np.max(neffApass)],[maxNeffRave, maxNeffRave], color='black', alpha=0.1)
        fig.colorbar(points)
        ax.set_xlabel('Apass Neff')
        ax.set_ylabel('Rave Neff')
        plt.tight_layout()
        fig.savefig('neffApassVsTgas.png')

    #spot check dwarfs with lots of Apass neighbors but few Rave neighbors
    manyApassfewRave = np.where((neffRave < maxNeffRave) & (neffApass > minNeffApass))[0]
    manyRavefewApass = np.where((neffRave > minNeffRave) & (neffApass < maxNeffApass))[0]
    print 'Lots of Apass, few Rave: ', manyApassfewRave
    print 'Few Apass, lots of Rave: ', manyRavefewApass

    try:
        data = np.load('gaussianArrays.npz')
        gaussAmplitudes = data['gaussAmplitudes']
        gaussMeans = data['gaussMeans']
        gaussCov = data['gaussCov']
        randarray = data['index']
    except IOError:
        nValues = 3 #logg, Teff, Fe/H
        ngauss = 2
        nstars = 5
        gaussAmplitudes = np.zeros((nstars, ngauss))
        gaussMeans = np.zeros((nstars, ngauss, nValues))
        gaussCov = np.zeros((nstars, ngauss, nValues, nValues))

        randarray = np.random.randint(0, high=len(apassCutMatched), size=nstars)
        for i, j in enumerate(randarray):
            gaussAmplitudes[i], gaussMeans[i], gaussCov[i] = XD(raveTwinIndex[j], raveCutMatched, chisqApass[j], ngauss=ngauss)
            np.savez('gaussianArrays.npz', gaussAmplitudes=gaussAmplitudes, gaussMeans=gaussMeans, gaussCov=gaussCov, index=randarray)


    fig, ax = plt.subplots(3)
    for i, (key, keyerror) in enumerate(zip(['TEFF', 'LOGG', 'FE_H'],['E_TEFF', 'E_LOGG', 'E_FE_H'])):
        mean = (gaussMeans[:,0,i]*gaussAmplitudes[:,0] + gaussMeans[:,1, i]*gaussAmplitudes[:,1])/(gaussAmplitudes[:,0] + gaussAmplitudes[:,1])
        cov = (gaussCov[:,0, i, i] + gaussCov[:,1,i,i])
        if key == 'TEFF':
            mean = 1000.*mean
            cov = 1000.*cov
        #ax[i].scatter(raveCutMatched[raveSourceIndex[randarray]][key], gaussMeans[:,0,i])
        ax[i].scatter(raveCutMatched[raveSourceIndex[randarray]][key], mean)
        ax[i].errorbar(raveCutMatched[raveSourceIndex[randarray]][key], mean, xerr=raveCutMatched[raveSourceIndex[randarray]][keyerror], yerr=np.sqrt(cov), fmt="none", ecolor='black', zorder=0, lw=0.5, mew=0)
        xmin, xmax = ax[i].get_xlim()
        plotarray = np.linspace(xmin, xmax, 100)
        ax[i].plot(plotarray, plotarray)


    plt.show()

    #nplot = 10
    #randarray = np.random.randint(0, high=len(apassCutMatched), size=nplot)
    #plotComparisons(randarray, apassTwinIndex, apassSourceIndex, raveTwinIndex, raveSourceIndex, apassCutMatched, raveCutMatched, varmuCutMatched, chisqApass, nplot=nplot, filename='random.png', ngauss=2)
    #randarray = randarray[np.argsort(M_V[sources][randarray])

    #nplot = len(manyApassfewRave)
    #print 'The number of many Apass but few Rave neighbors: ', nplot
    #plotComparisons(manyApassfewRave, apassTwinIndex, apassSourceIndex, raveTwinIndex, raveSourceIndex, apassCutMatched, raveCutMatched, varmuCutMatched, chisqApass, nplot=nplot, filename='manyApassFewRave.png')


    #nplot = len(manyRavefewApass)
    #print 'The number of many Rave but few Apass neighbors: ', nplot
    #plotComparisons(manyRavefewApass, apassTwinIndex, apassSourceIndex, raveTwinIndex, raveSourceIndex, apassCutMatched, raveCutMatched, varmuCutMatched, chisqApass, nplot=nplot, filename='manyRavefewApass.png')
