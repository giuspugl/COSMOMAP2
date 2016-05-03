import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np

def test_coarse_operator():
    """
    Build and test the :class:`CoarseLO`.
    """
    nt,npix,nb= 400,20,1
    blocksize=nt/nb
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)
    c=bash_colors()
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():

        P=SparseLO(npix,nt,pairs,phi,pol)
        npix=P.ncols
        x0=np.zeros(pol*npix)
        N=BlockLO(blocksize,t,offdiag=True)
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
        b=P.T*N*d
        A=P.T*N*P
        diagN=lp.DiagonalOperator(diag*nt)

        tol=1.e-5
        B=BlockDiagonalLO(P,npix,pol=pol)
        #B=(P.T*P).to_array()
        eigv ,Z=spla.eigsh(A,M=B,Minv=Mbd,k=5,which='SM',ncv=15,tol=tol)
        r=Z.shape[1]
        Zd=DeflationLO(Z)
        # Build Coarse operator
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]
        invE=CoarseLO(Z,Az,r,apply='eig')
        E=dgemm(Z,Az.T)

        v=np.ones(r)
        y=invE*v
        v2=np.dot(E,invE*v)
        y2=la.solve(E,v)
        assert np.allclose(v,v2) and np.allclose(y2,y)
filter_warnings("ignore")
#test_coarse_operator()
