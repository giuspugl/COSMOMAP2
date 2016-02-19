import scipy.linalg as la
from interfaces import *
from utilities import *
#import scipy.sparse.linalg as spla
import numpy as np

def test_2level_preconditioner():
    """
    Build and test the expected algebraic properties
    of  the M2 level preconditioner.
    """
    filter_warnings("ignore")
    
    nt,npix,nb= 400,40,2
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)
    blocksize=nt/nb
    N=BlockLO(blocksize,t,offdiag=True)
    import time
    for pol in range(1,4):
        x0=np.zeros(pol*npix)
        P=SparseLO(npix,nt,pairs,phi,pol=pol,w=diag )
        diagN=BlockLO(blocksize,diag,offdiag=False)
        #M=InverseLO(P.T*diagN*P,method=spla.cg)
        M=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
        tol=1.e-4
        #print nb,nt,npix
        b=P.T*N*d
        A=P.T*N*P
        B=P.T*P
        # Build deflation supspace
        start=time.clock()
        eigv,Z=spla.eigs(A,k=6,which='SM',ncv=14,tol=tol)
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
        AZ=[]
        for i in Zd.z:
            AZ.append(A*i)

        for i in range(r):
            assert (np.allclose(M2*AZ[i],Zd.z[i]) and norm2(R*AZ[i])<=1.e-10)
            x,info=spla.cg(M2*A,Z[:,i],tol=tol,maxiter=2)
            assert info==0

test_2level_preconditioner()
