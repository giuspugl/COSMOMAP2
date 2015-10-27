import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np

def test_deflation_operator():
    """
    Build and test the deflation subspace matrix Z checking
    whether its columns are linearly independent <=> determ(Z.T*Z)!=0
    being Z.T*Z a r x r matrix, r =rank(Z)
    """

    nt,npix,nb,pol=400,200,2,1
    blocksize=nt/nb

    d,pairs,phi,t,diag,x0=system_setup(nt,npix,nb,pol)
    for i in range(2,4):
        tol=10**(-i)
        for j in range(2,3) :
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
            #rank(Z) must be equal to the size of the deflation subspace

            assert rank==r

            prod=dgemm(Z,Z.T)
            determ=la.det(prod)
            # the selected eigenvectors (columns of Z) has to
            # be linearly independent <=> determ(Z.T*Z)!=0 being a r x r matrix
            assert determ!=0

            Zd=DeflationLO(Z)
            v=np.ones(r)
            y2=Z.dot(v)
            y= Zd*v
            #print y2,y

            x=np.ones(npix)
            s2=Z.T.dot(x)
            s=Zd.T*x
            #print s2,s
            assert np.allclose(y2,y)
            assert np.allclose(s2,s)


def test_deflation_operator_pol():
    """
    Build and test the deflation subspace matrix Z checking
    whether its columns are linearly independent <=> determ(Z.T*Z)!=0
    being Z.T*Z a r x r matrix, r =rank(Z)
    POLARIZATION CASE
    """
    nt,npix,nb,pol=200,30,2,3
    blocksize=nt/nb

    d,pairs,phi,t,diag,x0=system_setup(nt,npix,nb,pol)
    for i in range(1,3):
        tol=10**(-i)
        for j in range(1,2) :
            eps=10**(-j)
            N=BlockLO(blocksize,t,offdiag=True)
            P=SparseLO(npix,nt,pairs,phi,pol)
            diagN=BlockLO(blocksize,diag,offdiag=False)
            M=InverseLO(P.T*diagN*P,method=spla.cg)

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

            Zd=DeflationLO(Z)
            v=np.ones(r)
            y2=Z.dot(v)
            y= Zd*v
            #print y2,y

            x=np.ones(pol*npix)
            s2=Z.T.dot(x)
            s=Zd.T*x
            #print s2,s
            assert np.allclose(y2,y)
            assert np.allclose(s2,s)



test_deflation_operator()
test_deflation_operator_pol()
