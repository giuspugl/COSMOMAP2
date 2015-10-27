import random as rd
import numpy as np
from scipy.linalg import get_blas_funcs
import math as m



def profile_run():
    import cProfile
    pr=cProfile.Profile()
    return pr

def output_profile(pr):
    import pstats,StringIO
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


def rescalepixels(pixs):
    minpix=min(pixs)
    maxpix=max(pixs)

    obspix=pixs - minpix
    return minpix,obspix,maxpix


def angles_gen(theta0,n,sample_freq=200. ,whwp_freq=2.5):
    """
    Generate  polarization angle given the sample frequency of the instrument,
    the frequency of HWP and the size ``n`` of the timestream.


    """
    #print theta0,sample_freq,whwp_freq,n
    return np.array([theta0+ 2*np.pi*whwp_freq/sample_freq*i for i in xrange(n)])

def count(ck):
    """
    iteration counter
    """
    ck+=1
    #print ck


def pairs_gen(nrows,ncols,pol=1):
    """
    Generate random ``int``s  to fill the pointing matrix for observed pixels.
    Implemented even for polarization runs.
    """
    if ncols<3:
        raise RuntimeError("Not enough pixels!\n Please set Npix >=3, you have set Npix=%d"%ncols)

    #pairs=[]
    #if pol==1:
        #for i in xrange(nrows):
            #pairs.append((i,rd.randint(0 ,ncols-1)))
            #pairs.append(rd.randint(0 ,ncols-1))
    js=np.random.randint(0,high=ncols,size=nrows)

    return js
    """
    elif pol==3:

        max_multi =3* m.floor(float(ncols) / 3)

        #for i in xrange(nrows):
        #    j=3*rd.randint(0 ,max_multi-1)
            #pairs.append((i,j))
        #    pairs.append(j)
        js=np.random.randint(0,high= max_multi-1,size=nrows)
        #print max_multi,js
        print js
        #js*=3.
        #return pairs
        return js
    """

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
        raise RuntimeError("convergence not achieved after %d iterations"%info)
        return False
    #    print '++++++++++++++++++++++++++++++++++++++'
    #    print '| convergence to tolerance not achieved after  |'
    #    print '| ', info,' iterations |'
    #    print '++++++++++++++++++++++++++++++++++++++'



def noise_val(nb,bandwidth=1):
    """
    Generate  elements to fill the  noise covariance
    matrix with a  random ditribution ``N_tt'=<n_t n_t'>``.

    **Parameters**
    - ``nb`` : {int}
        number of noise stationary intervals,
        i.e. number  of blocks in N_tt'.
    - ``bandwidth`` : {int}
        the width of the diagonal band.
        e.g. :
        -   ``bandwidth=1`` define the first up and low diagonal terms.
        -   ``bandwidth=2`` 2 off diagonal terms.

    **Returns**

    - ``t``: {list of arrays }
        ``shape=(nb,bandwidth)``
    - ``diag`` : {list }, ``size = nb``
        diagonal values of each block .
    """
    diag=[]
    t=[]
    for i in range(nb):
        t.append( np.random.random(size=bandwidth) )
    diag=[i[0] for i in t]
    return  t, diag


def system_setup(nt,npix,nb,pol=1):
    """
    Setup the linear system

    **Returns**
    - d :{array}
        a ``nt`` array of random numbers;
    - pairs: {list of tuples}
        the (i,j) non-null indices of the pointing matrix;
    - phi :{array}
        angles if ``pol=3``
    - t,diag :  {outputs of ``noise_val``}
        noise values to construct the noise covariance matrix
    - x : {array}
        the initial solution .
    """
    d=np.random.random(nt)
    pairs=pairs_gen(nt,npix,pol)
    phi=None
    if pol==3:
        phi=angles_gen(rd.uniform(0,np.pi),nt)
    bandsize=2
    t, diag=noise_val(nb,bandsize)
    x=np.zeros(pol*npix)
    return d,pairs,phi,t,diag,x
