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
            Zd=DeflationLO(Z)
            # Build Coarse operator
            E=CoarseLO(Z,A,r)
            #Build the 2-level preconditioner
            I= lp.IdentityOperator(pol*npix)

            R=I - A*Zd*E*Zd.T
            M2=M*R + Zd*E*Zd.T
            AZ=[]
            for i in Zd.z:
                AZ.append(A*i)

            for i in range(r):
                    assert (np.allclose(M2*AZ[i],Zd.z[i]) and norm2(R*AZ[i])<=1.e-10)

            x0=np.zeros(pol*npix)
            x,info=spla.gmres(A,b,x0=x0,tol=tol,maxiter=100,M=M2)
            assert info==0




def test_2level_preconditioner_pol():
    """
    Build and test the expected algebraic properties
    of  the M2 level preconditioner.
    """
    nt,npix,nb,pol= 100,10,2,3
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
            Zd=DeflationLO(Z)
            # Build Coarse operator
            E=CoarseLO(Z,A,r)
            #Build the 2-level preconditioner
            I= lp.IdentityOperator(pol*npix)

            R=I - A*Zd*E*Zd.T
            M2=M*R + Zd*E*Zd.T
            AZ=[]
            for i in Zd.z:
                AZ.append(A*i)

            for i in range(r):
                    assert (np.allclose(M2*AZ[i],Zd.z[i]) and norm2(R*AZ[i])<=1.e-10)

            x0=np.zeros(pol*npix)
            x,info=spla.gmres(A,b,x0=x0,tol=tol,maxiter=100,M=M2)
            assert info==0

test_2level_preconditioner()
test_2level_preconditioner_pol()
