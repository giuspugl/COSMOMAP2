import random as rd
import numpy as np
from scipy.linalg import get_blas_funcs
import math as m



def profile_run():
    """
    Profile the execution with :mod:`cProfile`
    """
    import cProfile
    pr=cProfile.Profile()
    return pr

def output_profile(pr):
    """
    Output of the profiling with :func:`profile_run`.

    **Parameter**

    - ``pr``:
        instance returned by :func:`profile_run`

    """
    import pstats,StringIO
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
    pass

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



def pairs_gen(nrows,ncols,pol=1):
    """
    Generate random ``int`` numbers   to fill the pointing matrix for observed pixels.
    Implemented even for polarization runs.

    """
    if ncols<3:
        raise RuntimeError("Not enough pixels!\n Please set Npix >=3, you have set Npix=%d"%ncols)

    js=np.random.randint(0,high=ncols,size=nrows)

    return js


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
    matrix with a  random ditribution :math:`N_{tt}= < n_t n_t >`.

    **Parameters**

    - ``nb`` : {int}
        number of noise stationary intervals,
        i.e. number  of blocks in N_tt'.
    - ``bandwidth`` : {int}
        the width of the diagonal band,e.g. :

        - ``bandwidth=1`` define the first up and low diagonal terms
        - ``bandwidth=2`` 2 off diagonal terms.

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

def subscan_resize(data,subscan):
    """
    Resize a tod-size array  by considering only the subscan intervals.
    """
    tmp=[]
    for i in xrange(len(subscan[0])):
        start=(subscan[1][i])
        end=subscan[1][i] + subscan[0][i]
        tmp.append(data[start:end])
    return np.concatenate(tmp)

def system_setup(nt,npix,nb,pol=1):
    """
    Setup the linear system

    **Returns**

    - ``d`` :{array}
        a ``nt`` array of random numbers;
    - ``pairs``: {array }
        the non-null indices of the pointing matrix;
    - phi :{array}
        angles if ``pol=3``
    - t,diag :  {outputs of :func:`noise_val`}
        noise values to construct the noise covariance matrix

    """
    d=np.random.random(nt)
    pairs=pairs_gen(nt,npix,pol)
    phi=None
    if pol==3:
        phi=angles_gen(rd.uniform(0,np.pi),nt)
    bandsize=2
    t, diag=noise_val(nb,bandsize)

    return d,pairs,phi,t,diag
