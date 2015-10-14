import random as rd
import numpy as np
from scipy.linalg import get_blas_funcs
import math as m

def scalprod(a,b):
    dot=get_blas_funcs('dot', (a,b))
    return dot(a,b)


def angles_gen(theta0,n,sample_freq=200. ,whwp_freq=2.5):
    """
    Generate  polarization angle given the sample frequency of the instrument,
    the frequency of HWP and the size ``n`` of the timestream.


    """
    #print theta0,sample_freq,whwp_freq,n
    return [theta0+ 2*np.pi*whwp_freq/sample_freq*i for i in xrange(n)]

def pairs_gen(nrows,ncols,pol=1):
    """
    Generate pairs (i,j) to fill the pointing matrix
    Implemented even for polarization runs.
    """
    if ncols<3:
        raise RuntimeError("Not enough pixels!\n Please set Npix >=3, you have set Npix=%d"%ncols)

    pairs=[]
    if pol==1:
        for i in xrange(nrows):
            pairs.append((i,rd.randint(0 ,ncols-1)))
        return pairs
    elif pol==3:
        max_multi =3* m.floor(float(ncols) / 3)
        for i in xrange(nrows):
            j=3*rd.randint(0 ,max_multi-1)
            pairs.append((i,j))

        return pairs


def checking_output(info):
    if info==0:
        return True
    #    print '+++++++++++++++++++++++++'
    #    print "| successful convergence |"
    #    print '+++++++++++++++++++++++++'

    if info<0:
        raise RuntimeError("illegal input or breakdown during the execution")
        return False
    #    print '+++++++++++++++++++++++++'
    #    print '| illegal input or breakdown |'
    #    print '+++++++++++++++++++++++++'
    elif info >0 :
        raise RuntimeError("convergence not achieved after %d iterations")%info
        return False
    #    print '++++++++++++++++++++++++++++++++++++++'
    #    print '| convergence to tolerance not achieved after  |'
    #    print '| ', info,' iterations |'
    #    print '++++++++++++++++++++++++++++++++++++++'



def noise_val(nb,bandwidth=1):
    """
        Generate  elements to fill the  noise covariance
        matrix with a  random ditribution ``N_tt'=<n_t n_t'>``.

        Parameters
        ----------

        ``nb`` : {int}
            number of noise stationary intervals,
            i.e. number  of blocks in N_tt'.
        ``bandwidth`` : {int}
            the width of the diagonal band.
            e.g. :
            -   ``bandwidth=1`` define the first up and low diagonal terms.
            -   ``bandwidth=2`` 2 off diagonal terms.

        Returns
        -------
        ``t``: {list of arrays }
            ``shape=[nb,bandwidth]``
        ``diag`` : {list }, ``size = nb``
                diagonal values of each block .
    """
    diag=[]
    t=[]
    for i in range(nb):
        t.append( np.random.random(size=bandwidth) )
    diag=[i[0] for i in t]
    return  t, diag
