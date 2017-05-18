import numpy as np
import matplotlib
matplotlib.use('pdf')
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from astroML.plotting.tools import draw_ellipse
from astroML.plotting import setup_text_plots
from sklearn.mixture import GMM as skl_GMM
import drawEllipse
import matplotlib as mpl
plt.style.use('seaborn-talk')
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
from scipy.stats import gaussian_kde
import corner

def plot_bic(param_range,bics,lowest_comp):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 6))
    plt.bar(param_range-0.25,bics,color='blue',width=0.5)
    plt.text(lowest_comp, bics.min() * 0.97 + .03 * bics.max(), '*',
             fontsize=14, ha='center')

    plt.xticks(param_range)
    plt.ylim(bics.min() - 0.01 * (bics.max() - bics.min()),
             bics.max() + 0.01 * (bics.max() - bics.min()))
    plt.xlim(param_range.min() - 1, param_range.max() + 1)

    plt.xticks(param_range,fontsize=14)
    plt.yticks(fontsize=14)


    plt.xlabel('Number of components',fontsize=18)
    plt.ylabel('BIC score',fontsize=18)

    plt.show()

def plot_val_curve(param_range, train_mean, train_std, test_mean,
                   test_std, log=False):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig=plt.figure(figsize=(12,8))
    plt.plot(param_range, train_mean, label="Training score",
             color="red")
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.2, color="red")
    plt.plot(param_range, test_mean,label="Cross-validation score",
             color="green")
    plt.fill_between(param_range, test_mean - test_std,
                     test_mean + test_std, alpha=0.2, color="green")

    plt.legend(loc="best")
    plt.xlabel("Number of Components", fontsize=18)
    plt.ylabel("Score", fontsize=18)
    plt.xlim(param_range.min(),param_range.max())
    if log: plt.xscale('log', basex=2)
    plt.savefig('val_curve.png')


def absMagKinda2absMag(absMagKinda):
    """
    convert my funny units of parallax[mas]*10**(0.2*apparent magnitude[mag]) to an absolute magnitude [mag]
    """
    absMagKinda_in_arcseconds = absMagKinda/1e3 #first convert parallax from mas ==> arcseconds
    return 5.*np.log10(10.*absMagKinda_in_arcseconds)

def kdeDensity(ax, x, y, threshold=0.01, bins=100, s=1, lw=0, alpha=1):

    points = np.vstack([x, y])

    # perform kernel density estimate
    kde = gaussian_kde(points)
    z = kde(points)

    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, x)
    y = np.ma.masked_where(z > threshold, y)

    # plot unmasked points
    ax.scatter(x, y, c='black', marker='.', s=s, lw=lw, alpha=alpha)

    # get bounds from axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # prepare grid for density map
    xedges = np.linspace(xmin, xmax, bins)
    yedges = np.linspace(ymin, ymax, bins)
    xx, yy = np.meshgrid(xedges, yedges)
    gridpoints = np.array([xx.ravel(), yy.ravel()])

    # compute density map
    zz = np.reshape(kde(gridpoints), xx.shape)

    # plot density map
    im = ax.imshow(zz, cmap='Greys', interpolation='nearest',
               origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax])

    # plot threshold contour
    cs = ax.contour(xx, yy, zz, levels=[threshold], colors='black', linestyle='--', linewidths=0.5)

    # show
    #fig.colorbar(im)
    return ax


def plot_sample(x, y, samplex, sampley, xdgmm, xlabel='x', ylabel='y', xerr=None, yerr=None, ylim=(6, -6), xlim=(0.5, 1.5), errSubsample=1.2e6, thresholdScatter=0.1, binsScatter=200, c='black',  norm=None, cmap=None, contourColor='k', posterior=False, ind=None, plot_contours=True, sdss5=False):

    prng = np.random.RandomState(1234567890)
    setup_text_plots(fontsize=16, usetex=True)
    plt.clf()
    alpha = 0.1
    alpha_points = 0.01
    figData = plt.figure(figsize=(12, 5.5))
    figPrior = plt.figure(figsize=(12, 5.5))
    for fig in [figData, figPrior]:
        fig.subplots_adjust(left=0.1, right=0.95,
                            bottom=0.15, top=0.95,
                            wspace=0.1, hspace=0.1)

    ax1 = figData.add_subplot(121)
    levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
    cNorm  = plt.matplotlib.colors.LogNorm(vmin=3, vmax=1e5)
    #ax1.hist2d(x, y, bins=100, norm=cNorm, cmap='Greys')

    if sdss5: plot_contours=False
    im = corner.hist2d(x, y, ax=ax1, levels=levels, bins=200, no_fill_contours=True, plot_density=False, color=contourColor, rasterized=rasterized, plot_contours=plot_contours)

    #im = ax1.scatter(x, y, s=1, lw=0, c=c, alpha=alpha, norm=norm, cmap=cmap)

    ax2 = figData.add_subplot(122)
    if ind is None: ind = prng.randint(0, len(x), size=errSubsample)
    #ax2.scatter(x[ind], y[ind], s=1, lw=0, c='black', alpha=alpha_points)
    ax2.errorbar(x[ind], y[ind], xerr=xerr[ind], yerr=[yerr[0][ind], yerr[1][ind]], fmt="none", zorder=0, mew=0, ecolor='black', alpha=0.5, elinewidth=0.5)

    ax3 = figPrior.add_subplot(121)
    #ax3.hist2d(x, y, bins=100, norm=cNorm, cmap='Greys')
    #kdeDensity(ax3, samplex, sampley, threshold=thresholdScatter, bins=binsScatter, s=1, lw=0, alpha=alpha)
    corner.hist2d(samplex, sampley, ax=ax3, levels=levels, bins=200, no_fill_contours=True, plot_density=False, color=contourColor, rasterized=rasterized, plot_contours=False)
    ax3.scatter(samplex, sampley, s=1, lw=0, c='k', alpha=alpha)

    ax4 = figPrior.add_subplot(122)
    for i in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[i], xdgmm.V[i])
        ax4.plot(points[0, :], absMagKinda2absMag(points[1,:]), 'k-', alpha=xdgmm.weights[i]/np.max(xdgmm.weights))
        #draw_ellipse(xdgmm.mu[i], xdgmm.V[i], scales=[2], ax=ax4,
        #         ec='None', fc='gray', alpha=xdgmm.weights[i]/np.max(xdgmm.weights)*0.1)




    #xlim = ax4.get_xlim()
    #ylim = ylim #ax3.get_ylim()


    titles = ["Observed Distribution", "Obs+Noise Distribution",
              "Extreme Deconvolution\n  resampling",
              "Extreme Deconvolution\n  cluster locations"]
    if posterior:
        titles = ["De-noised Expectation Values", "Posterior Distributions",
                  "Extreme Deconvolution\n  resampling",
                  "Extreme Deconvolution\n  cluster locations"]
    if sdss5:
        titles=['','','','']
    ax = [ax1, ax2, ax3, ax4]

    for i in range(4):
        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylim[0], ylim[1]*1.1)

        #ax[i].xaxis.set_major_locator(plt.MultipleLocator([-1, 0, 1]))
        #ax[i].yaxis.set_major_locator(plt.MultipleLocator([3, 4, 5, 6]))

        ax[i].text(0.05, 0.95, titles[i],
                   ha='left', va='top', transform=ax[i].transAxes, fontsize=18)

        #if i in (0, 1):
        #    ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        #else:
        ax[i].set_xlabel(xlabel, fontsize = 18)

        if i in (1, 3):
            ax[i].yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax[i].set_ylabel(ylabel, fontsize = 18)

    #ax[3].text(0.05, 0.95, titles[3],
    #               ha='left', va='top', transform=ax[3].transAxes)

    #ax[3].set_ylabel(r'$\varpi10^{0.2*m_G}$', fontsize=18)
    #ax[3].set_xlim(-2, 3)
    #ax[3].set_ylim(3, -1)
    #ax[3].yaxis.tick_right()
    #ax[3].yaxis.set_label_position("right")
    #plt.tight_layout()
    """
    if norm is not None:
        figData.subplots_adjust(left=0.2, right=0.95)
        cbar_ax = figData.add_axes([0.01, 0.125, 0.02, 0.75])
        cb = figData.colorbar(im, cax=cbar_ax)
        #cb = plt.colorbar(im, ax=axes[2])
        cb.set_label(r'$ln \, \tilde{\sigma}_{\varpi}^2 - ln \, \sigma_{\varpi}^2$', fontsize=20)
        cb.set_clim(-7, 2)
    """

    figData.savefig('plot_sample.data.pdf', format='pdf')
    figPrior.savefig('plot_sample.prior.pdf', format='pdf')




def plot_cond_model(xdgmm, cond_xdgmm, y):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(111)
    for i in range(xdgmm.n_components):
        draw_ellipse(xdgmm.mu[i], xdgmm.V[i], scales=[2], ax=ax1,
                     ec='None', fc='gray', alpha=0.2)

    ax1.plot([-2,15],[y,y],color='blue',linewidth=2)
    ax1.set_xlim(-1, 13)
    ax1.set_ylim(-6, 16)
    ax1.set_xlabel('$x$', fontsize = 18)
    ax1.set_ylabel('$y$', fontsize = 18)

    ax2 = ax1.twinx()
    x = np.array([np.linspace(-2,14,1000)]).T

    gmm=skl_GMM(n_components = cond_xdgmm.n_components,
                covariance_type = 'full')
    gmm.means_ = cond_xdgmm.mu
    gmm.weights_ = cond_xdgmm.weights
    gmm.covars_ = cond_xdgmm.V

    logprob, responsibilities = gmm.score_samples(x)

    pdf = np.exp(logprob)
    ax2.plot(x, pdf, color='red', linewidth = 2,
             label='Cond. dist. of $x$ given $y='+str(y)+'\pm 0.05$')
    ax2.legend()
    ax2.set_ylabel('Probability', fontsize= 18 )
    ax2.set_ylim(0, 0.52)
    ax1.set_xlim(-1, 13)
    plt.show()

def plot_cond_sample(x, y):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 9))

    plt.hist(x, 50, histtype='step', color='red',lw=2)

    plt.ylim(0,70)
    plt.xlim(-1,13)

    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('Number of Points', fontsize=18)

    plt.show()

def plot_conditional_predictions(y, true_x, predicted_x):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 9))

    plt.scatter(true_x, y, color='red', s=4, marker='o',
                label="True Distribution")
    plt.scatter(predicted_x, y, color='blue', s=4, marker='o',
                label="Predicted Distribution")

    plt.xlim(-1, 13)
    plt.ylim(-6, 16)
    plt.legend(loc=2, scatterpoints=1)

    plt.xlabel('$x$', fontsize = 18)
    plt.ylabel('$y$', fontsize = 18)
    plt.show()
