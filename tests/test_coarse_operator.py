import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np

def test_coarse_operator():
    """
    Build and test the coarse operator E.
    """
    nt,npix,nb,pol= 100,30,2,1
    blocksize=nt/nb
    d,pairs,phi,t,diag,x0=system_setup(nt,npix,nb,pol)
    blocksize=nt/nb
    for i in range(2,4):
        tol=10**(-i)
        for j in range(1,2) :
            eps=10**(-j)
            P=SparseLO(npix,nt,pairs,phi,pol)
            N=BlockLO(blocksize,t,offdiag=True)
            diagN=BlockLO(blocksize,diag,offdiag=False)
            M=InverseLO(P.T*diagN*P,method=spla.cg)
            #print nb,nt,npix
            b=P.T*N*d
            A=P.T*N*P
            # Build deflation supspace
            h=[]
            w=[]
            w,h=arnoldi(M*A,b,x0=x0,tol=tol,maxiter=1,inner_m=pol*npix)
            m=len(w)
            H=build_hess(h,m)

            z,y=la.eig(H,check_finite=False)

            Z,r= build_Z(z,y, w, eps)

            # Build Coarse operator

            E=CoarseLO(Z,A,r)

            v=np.ones(r)
            y=E*v

            y2= la.solve(E.L.dot(E.U),v)
            assert  np.allclose(y2,y)
            
test_coarse_operator()
