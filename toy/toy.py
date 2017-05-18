import numpy as np
import scipy.optimize as op
import sys


def make_fake_data(parsTrue, N=512):
    np.random.seed(42)
    mtrue, btrue, ttrue = parsTrue
    xns = np.random.uniform(size=N) * 2. - 1.
    tns = np.random.normal(scale=ttrue, size=N)
    #tns = np.random.normal(size=N) * ttrue
    yntrues = mtrue * xns + btrue + tns
    sigmans = (np.random.uniform(size=N)*1.5) ** 3.
    #yns = yntrues + np.random.normal(size=N) * sigmans
    yns = yntrues + np.random.normal(scale=sigmans)
    fig, ax = plt.subplots(1,3)
    ax[0].scatter(xns, tns, s=2)
    ax[0].set_ylabel('sampled noise from ttrue gaussian')
    ax[1].scatter(xns, sigmans, s=2)
    ax[1].set_ylabel('sigmans')
    ax[2].scatter(xns, np.random.normal(scale=sigmans), s=2)
    ax[2].set_ylabel('sampled noise from sigmans gaussian')
    plt.tight_layout()
    fig.savefig('laurenUnderstanding.png')
    return xns, yns, sigmans, yntrues

def objective(pars, xns, yns, sigmas):
    m, b, t = pars
    resids = yns - (m * xns + b)
    return np.sum(resids * resids / (sigmas * sigmas + t * t)
                  + np.log(sigmas * sigmas + t * t))

def get_best_fit_pars(xns, yns, sigmans):
    pars0 = np.array([-1.5, 0., 1.])
    result = op.minimize(objective, pars0, args=(xns, yns, sigmans))
    return result.x

def denoise_one_datum(xn, yn, sigman, m, b, t):
    s2inv = 1. / (sigman * sigman)
    t2inv = 1. / (t * t)
    return (yn * s2inv + (m * xn + b) * t2inv) / (s2inv + t2inv), \
        np.sqrt(1. / (s2inv + t2inv))

def plot(fig, axes, figNoise, axesNoise, xns, yns, sigmans, yntrues, m, b, t, mtrue, btrue, ttrue, ydns, sigmadns, nexamples=5):

    alpha_all = 0.05
    alpha_chosen = 1.0
    xlim = (-1, 1)
    ylim = (-5, 5)

    dataMap = mpl.cm.get_cmap('Blues')
    dataColor = dataMap(0.75)
    trueMap = mpl.cm.get_cmap('Reds')
    trueColor = trueMap(0.75)

    for ax in axes[1:-1]:
        ax.errorbar(xns, yns, yerr=sigmans, fmt="o", color="k", alpha=alpha_all ,mew=0)
        ax.errorbar(xns[0:nexamples], yns[0:nexamples], yerr=sigmans[0:nexamples], fmt="o", color="k", zorder=37, alpha=alpha_chosen, mew=0)

    xp = np.array(xlim)
    axes[0].plot(xp, mtrue*xp + btrue + ttrue, color='red', linewidth=2, alpha=0.75, label=r'$y=m_{true}\,x+b_{true}\pm t$')
    axes[0].plot(xp, mtrue*xp + btrue - ttrue, color='red', linewidth=2, alpha=0.75)
    axes[0].scatter(xns, yntrues, c='red', lw=0, alpha=0.5, label=r'$y_{true,n}$')
    axes[0].legend(loc='best', fontsize=15)

    axes[2].plot(xp, m * xp + b + t, color=dataColor)
    axes[2].plot(xp, m * xp + b - t, color=dataColor)
    axes[2].scatter(xns[0:nexamples], yntrues[0:nexamples], c=trueColor, lw=2, zorder=36, alpha=alpha_chosen, facecolors='None')
    axes[2].plot(xp, mtrue*xp + btrue + ttrue, color=trueColor, zorder=35)
    axes[2].plot(xp, mtrue*xp + btrue - ttrue, color=trueColor, zorder=34)
    r1 = axes[2].add_patch(mpl.patches.Rectangle((-10,-10), 0.1, 0.1, color='black', alpha=alpha_chosen))
    r2 = axes[2].add_patch(mpl.patches.Rectangle((-10,-10), 0.1, 0.1, color=trueColor, alpha=alpha_chosen))
    r3 = axes[2].add_patch(mpl.patches.Rectangle((-10,-10), 0.1, 0.1, color=dataColor, alpha=alpha_chosen))
    axes[2].legend((r1,r2,r3), ('data', 'truth', 'denoised'), loc='best', fontsize=12)

    axes[2].errorbar(xns[0:nexamples], ydns[0:nexamples], yerr=sigmadns[0:nexamples], fmt="o", color=dataColor, zorder=37, alpha=alpha_chosen, mew=0)

    norm = mpl.colors.Normalize(vmin=0, vmax=9)
    im = axes[3].scatter(xns,  ydns,  c=sigmans**2., cmap='Blues', norm=norm, alpha=0.5, lw=0)
    fig.subplots_adjust(left=0.1, right=0.89)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.35])
    cb = fig.colorbar(im, cax=cbar_ax)
    #cb = plt.colorbar(im, ax=axes[2])
    cb.set_label(r'$\sigma_n^2$', fontsize=20)
    cb.set_clim(-4, 9)


    axes[3].errorbar(xns, ydns, yerr=sigmadns, fmt="None", mew=0, color='black', alpha=0.25, elinewidth=0.5)
    #axes[2].errorbar(xns[0:nexamples], ydns[0:nexamples], yerr=sigmadns[0:nexamples], fmt="o", color="b", zorder=37, alpha=alpha_chosen, mew=0)
    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks((-1, -0.5, 0, 0.5, 1))
        ax.set_xticklabels((-1, 0.5, 0, 0.5, 1))
    #plt.tight_layout()

    axesNoise[0].hist(sigmans, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[0].set_xlabel(r'$\sigma_n$', fontsize=15)
    axesNoise[1].hist(sigmadns, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[1].set_xlabel(r'$\tilde\sigma_n$', fontsize=15)
    for a in axesNoise[0:2]:
        a.axvline(t, label='$t$', color='black', lw=2)
        a.legend(fontsize=15, loc='best')
    axesNoise[2].hist(yns-yntrues, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[2].set_xlabel(r'$y_n - y_{true,n}$', fontsize=15)
    axesNoise[3].hist(ydns-yntrues, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[3].set_xlabel(r'$<p(y_{true,n})> - y_{true,n}$', fontsize=15)
    #plt.tight_layout()

    return fig, axes, figNoise, axesNoise

def gaussian(mean, sigma, array, amplitude=1.0):
    return amplitude/np.sqrt(2.*np.pi*sigma**2.)*np.exp(-(array - mean)**2./(2.*sigma**2.))

def exampleParallax():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-talk')
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    x = np.linspace(-2., 8., 1000)
    positive = x > 0
    parallax = 0.5
    sigma_parallax = [0.1, 0.2, 0.3]
    linestyle = ['-', '--', ':']
    labelParallax = r'$P(\varpi \,|\, \varpi_{true}, \sigma_{\varpi} )$'
    labelDistance = r'$P(\frac{1}{\varpi} \,|\, \varpi_{true}, \sigma_{\varpi} )$'
    for i, (sigma, ls) in enumerate(zip(sigma_parallax, linestyle)):
        likelihood = gaussian(parallax, sigma, x)
        label = r'$\varpi/\sigma_{\varpi}=$' + '{0:.1f}'.format(parallax/sigma)
        ax[0].plot(x, likelihood, lw=2, label='likelihood', linestyle=ls)
        ax[1].plot(1./x[positive], likelihood[positive], lw=2, linestyle=ls, label=label)
        ax[0].set_xlabel('parallax [mas]', fontsize=18)
        ax[1].set_xlabel('distance [kpc]', fontsize=18)
        ax[0].set_ylabel(labelParallax, fontsize=18)
        ax[1].set_ylabel(labelDistance, fontsize=18)
        ax[0].set_xlim(-1, 3)
        ax[1].set_xlim(0, 8)
        ax[1].legend(loc='best', fontsize=15)
        #ax[0].legend(loc='best')
        fig.savefig('likelihoodExample' + str(i) + '.png')


    #fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    parallax = -0.25
    sigma = 0.3
    likelihood = gaussian(parallax, sigma, x)
    label = r'$\varpi/\sigma_{\varpi}=$' + '{0:.1f}'.format(np.abs(parallax)/sigma)
    ax[0].plot(x, likelihood, lw=2, label='likelihood')
    ax[1].plot(1./x[positive], likelihood[positive], lw=2, label=label)
    ax[0].set_xlim(-1, 3)
    ax[1].set_xlim(0, 8)
    ax[1].legend(loc='best', fontsize=15)
    #ax[0].legend(loc='best')
    #ax[0].set_xlabel('parallax [mas]', fontsize=18)
    #ax[1].set_xlabel('distance [kpc]', fontsize=18)
    #ax[1].set_xlim(0, 8)
    fig.savefig('likelihoodExampleNegative.png')
    plt.close(fig)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    xlim = (-1, 1)
    ylim = (-5, 5)

    nexamples =  np.int(sys.argv[1])
    mtrue, btrue, ttrue = -1.37, 0.2, 0.8
    parsTrue = [mtrue, btrue, ttrue] #m, b, t

    xns, yns, sigmans, yntrues = make_fake_data(parsTrue)
    m, b, t = get_best_fit_pars(xns, yns, sigmans)

    print m, b, t

    ydns = np.zeros_like(yns)
    sigmadns = np.zeros_like(sigmans)
    for n, (xn, yn, sigman) in enumerate(zip(xns, yns, sigmans)):
        ydns[n], sigmadns[n] = denoise_one_datum(xn, yn, sigman, m, b, t)

    for label, style in zip(['paper', 'talk'],['seaborn-paper', 'seaborn-talk']):

        plt.style.use(style)
        mpl.rcParams['xtick.labelsize'] = 18
        mpl.rcParams['ytick.labelsize'] = 18
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes = axes.flatten()
        figNoise, axesNoise = plt.subplots(2,2, figsize=(8,8))
        axesNoise = axesNoise.flatten()

        figTrue, axesTrue = plt.subplots(1, 2, figsize=(15, 5))
        xp = np.array(xlim)
        for ax in axesTrue:
            ax.plot(xp, mtrue*xp + btrue + ttrue, color='red', linewidth=2, alpha=0.75, label=r'$y=m_{true}\,x+b_{true}\pm t$')
            ax.plot(xp, mtrue*xp + btrue - ttrue, color='red', linewidth=2, alpha=0.75)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        axesTrue[0].scatter(xns, yntrues, c='red', lw=0, alpha=0.5, label=r'$y_{true,n}$: true y values')
        axesTrue[1].scatter(xns, yns, c='black', lw=0, alpha=0.5, label=r'$y_n$: noisy y values')
        axesTrue[1].errorbar(xns, yns, yerr=sigmans, c='black', fmt='None', alpha=0.5)
        axesTrue[0].legend(loc='best', fontsize=15)
        axesTrue[1].legend(loc='best', fontsize=15)
        figTrue.tight_layout()
        figTrue.savefig('toyTrue.png')

        fig, axes, figNoise, axesNoise = plot(fig, axes, figNoise, axesNoise, xns, yns, sigmans, yntrues, m, b, t, mtrue, btrue, ttrue, ydns, sigmadns, nexamples=nexamples)
        figNoise.tight_layout()
        figNoise.savefig('toyNoise.' + label + '.png')
        #fig.tight_layout()
        fig.savefig("toy." + label + ".png")
