import time
import scipy.linalg as la
from interfaces import *
from utilities import *
#import scipy.sparse.linalg as spla
import numpy as np

def test_2level_preconditioner():
    """
    Build and test the expected algebraic properties of  the M2 level preconditioner.
    """

    nt,npix,nb= 500,40,1
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)
    blocksize=nt/nb
    N=BlockLO(blocksize,t,offdiag=True)
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        npix     =  40
        processd =  ProcessTimeSamples(pairs,npix,pol=pol ,phi=phi)
        npix=   processd.get_new_pixel[0]
        P   =   SparseLO(npix,nt,pairs,pol=pol,angle_processed=processd)
        x0  =   np.zeros(pol*npix)
        M =   BlockDiagonalPreconditionerLO(processd ,npix,pol=pol)
        B   =   BlockDiagonalLO(processd,npix,pol=pol)
        x0=np.ones(pol*npix)
        tol=1.e-4
        b=P.T*N*d
        A=P.T*N*P
        # Build deflation supspace

        start=time.clock()
        eigv ,Z=spla.eigsh(A,M=B,Minv=M,k=5,v0=x0,which='SM',ncv=15,tol=tol)
        end=time.clock()

        r=Z.shape[1]
        # Build Coarse operator
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]
        E=CoarseLO(Z,Az,r)
        Zd=DeflationLO(Z)

        #Build the 2-level preconditioner
        I= lp.IdentityOperator(pol*npix)

        R=I - A*Zd*E*Zd.T
        M2=M*R + Zd*E*Zd.T

        for i in range(r):
            assert (np.allclose(M2*A*Z[:,i],Z[:,i]) and norm2(R*A*Z[:,i])<=1.e-10)
            x,info=spla.cg(M2*A,Z[:,i],tol=tol,maxiter=2)
            assert info==0
filter_warnings("ignore")
#test_2level_preconditioner()
