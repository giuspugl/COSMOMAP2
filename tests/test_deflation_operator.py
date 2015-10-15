import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np

def test_deflation_operator():
    """
    Build and test the deflation subspace matrix Z checking
    whether its columns are linearly independent.
    """

    nt,npix,nb=100,50,2
    blocksize=nt/nb
    for pol in [1,3]:
        d,pairs,phi,t,diag,x0=system_setup(nt,npix,nb,pol)
        for i in range(1,2):
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
                rank= np.linalg.matrix_rank(Z)
                assert rank==r
                prod=dgemm(Z,Z.T)
                determ=la.det(prod)
                assert determ!=0


test_deflation_operator()
