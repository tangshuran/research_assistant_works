import scipy as sp
from scipy.interpolate import interp1d 

def ECDF(seq):
    """
    Calculate the Empirical Cumuated Distribution Function (ecdf) from a sequence 'seq'.

    A scipy interpolation object is returned.
    """
    N=len(seq)
    sseq=sp.sort(seq)
    ecdf=sp.linspace(1./N,1,N)
    return interp1d(sseq,ecdf,bounds_error=False)

def ECDF2(seq):
    """
    Calculate the Empirical Cumuated Distribution Function (ecdf) from a sequence 'seq'.
    """
    N=len(seq)
    sseq=sp.sort(seq)
    ranks = sp.stats.rankdata(sseq)
    ecdf=ranks/(N+1)
    return ecdf


if __name__ == "__main__":
    import scipy as sp
    import scipy.stats as stats
    import pylab as pl
    N=100 # numper of points
    dist=stats.norm(loc=10.0, scale=2) # a 'frozen' distribution
    seq=dist.rvs(size=N) # random sample from the dist
    ecdf=ECDF(seq) # calculate the ecdf
    print ecdf
    x=sp.arange(0,20,0.1) # x-Values for the plot
    y=ecdf(x) # ecdf at these points
    truey=dist.cdf(x) # the cdf of the frozen dist (true cdf)
    pl.plot(x,y, x, truey) # plot ecdf and cdf
    pl.plot(sorted(seq), ECDF2(seq), '--')
    pl.grid(True)
    pl.show()

    
