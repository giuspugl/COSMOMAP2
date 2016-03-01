import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla

def test_preconditioner_times_matrix_gives_identity():
    """
    test  the action of  ``v=(A^T*A)^-1 * A^T*A * x`` onto  a vector x.
    check whther x==v.

    """
    filter_warnings("ignore")
    runcase={'IQU':3,'I':1,'QU':2}
    for pol in runcase.values():
        d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol,npairs=2)
        nt,npix,nb=len(d),len(hp_pixs),len(t)
        P=SparseLO(npix,nt,pixs,phi,pol=pol)
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
        B=BlockDiagonalLO(P,npix,pol=pol)
        #print pol
        if pol==1:
            offset=0
            x=np.ones(npix)
            #Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
        elif pol==3:
            components={'I':[1.,0.,0.],'Q':[0.,1.,0.],'U':[0.,0.,1.]}
            comp=components['Q']
            x=np.array(comp*(npix))
            offset=1
        #    Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol=pol,\
        #                    sin2=P.sin2,cos2=P.cos2,sincos=P.sincos,cos=P.cosine,sin=P.sine)
        elif pol==2:
            components={'Q':[1,0],'U':[0,1]}
            comp=components['Q']
            x=np.array(comp*(npix))
            offset=0

        v=Mbd*B*x
        pixel_to_check=[ pol*i+offset for i in P.mask]
        #assert np.allclose(v[P.mask],x[P.mask])
        assert np.allclose(v[pixel_to_check],x[pixel_to_check])

def test_symmetry_and_positive_definiteness():
    """
    Test the action of the block diagonal preconditioner, defined as
    :math: `M_{BD}=(A^T diag(N^{-1}) A)^{-1}`
    with a realistic scanning strategy.
    """
    runcase={'IQU':3,'I':1,'QU':2}
    for pol in runcase.values():

        d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol,npairs=2)
        nt,npix,nb=len(d),len(hp_pixs),len(t)
        #print nt,npix,nb,len(hp_pixs[pixs])
        nside=128
        N=BlockLO(nt/nb,t)
        P=SparseLO(npix,nt,pixs,phi,pol=pol,w=N.diag )

        A=P.T*N*P
        randarray=np.random.rand(pol*npix)
        assert  np.allclose(A*randarray, A.T *randarray)
        assert scalprod(randarray,A*randarray)>0.

        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol)
        assert  np.allclose(Mbd*randarray, Mbd.T *randarray)
        assert scalprod(randarray,Mbd*randarray)>0.


test_symmetry_and_positive_definiteness()
test_preconditioner_times_matrix_gives_identity()
