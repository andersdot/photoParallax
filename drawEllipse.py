import numpy as	np

def fixAbsMag(x):
    return 5.*np.log10(10.*x)

def plotvector(mean, var, step=0.001):
    """
    mean, var should be *projected* to the 2-d space in which plotting is about to occur
    """
    assert mean.shape == (2,)
    assert var.shape ==	(2, 2)
    ts = np.arange(0, 2. * np.pi, step) #magic
    w, v = np.linalg.eigh(var)
    ps = np.sqrt(w[0]) * (v[:, 0])[:,None] * (np.cos(ts))[None, :] + \
         np.sqrt(w[1]) * (v[:, 1])[:,None] * (np.sin(ts))[None, :] + \
	     mean[:, None]
    return ps

if __name__ == "__main__":
    from xdgmm import XDGMM
    import pylab as plt

    xdgmm = XDGMM(filename='xdgmm.1028gauss.1.2M.fit')
    amps = xdgmm.weights
    mus = xdgmm.mu
    Vs = xdgmm.V

    plt.clf()
    for	amp, mean, var in zip(amps, mus, Vs):
        ps = plotvector(mean, var)
        plt.plot(ps[0,:], fixAbsMag(ps[1,:]), "k-", alpha=amp/np.max(amps))
    plt.xlim(-2, 3)
    plt.ylim(10, -6)
    plt.savefig("drawEllipse.png")
