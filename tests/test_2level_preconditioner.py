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
        x0=np.ones(pol*npix)
        P=SparseLO(npix,nt,pairs,phi,pol=pol )
        #M=InverseLO(P.T*diagN*P,method=spla.cg)
        M=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
        tol=1.e-4
        #print nb,nt,npix
        b=P.T*N*d
        A=P.T*N*P
        #B=(P.T*P).to_array()

        B=BlockDiagonalLO(P,npix,pol=pol)
        # Build deflation supspace

        start=time.clock()
        eigv ,Z=spla.eigsh(A,M=B,Minv=M,k=5,v0=x0,which='SM',ncv=15,tol=tol)
        #eigv,Z=spla.eigs(A,k=6,which='SM',ncv=14,tol=tol)
        end=time.clock()
        #print "time to eigenv: %g"%(end-start)

        r=Z.shape[1]
        #print r
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

#test_2level_preconditioner()
