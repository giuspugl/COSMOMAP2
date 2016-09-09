from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla
import numpy as np
def vecT_vec(size):
    v=np.random.random(size)
    v/=norm2(v)
    return  np.outer(v,v.T)

def test_arnoldi_algorithm():
    """
    test the scipy routine of ARPACK to get the smallest eigenvectors of a matrix A

    """
    size=500
    diag=1.*np.arange(1,size+1) #np.logspace(-1,2,num=size)
    D=np.diag(diag)
    prec=np.diag(1./diag)

    A=D

    val=[min(diag),max(diag),diag[size/3],diag[-size/3],diag[size/2]]
    for i in val:
        vTv=vecT_vec(size)
        A+=float(i)*vTv

    B=spla.aslinearoperator(dgemm(prec,A))
    b=prec.dot(np.random.random(size))
    Alo=spla.aslinearoperator(A)
    mbd=spla.aslinearoperator(prec)
    x0=np.ones(size)
    import time

    eigs,eigv=spla.eigsh(mbd*Alo,k=len(val),which='SM',ncv=24,tol=1.e-3)

    #eigs,eigv=spla.eigsh(A,M=D,Minv=prec,v0=x0,k=len(val),which='SM',ncv=24,tol=1.e-3)
    #eigs,eigv=spla.eigsh(A,M=D,Minv=mbd,v0=x0,k=len(val),which='SM',ncv=48,tol=1.e-3)
    e=time.clock()
    #print "sm",e-s
    #print eigs,np.allclose(eigs,eigs1)
    r=len(eigs)
    print eigs 
    for i in range(r):
        print np.allclose(norm2(B*eigv[:,i])/norm2(eigv[:,i]),eigs[i])
        #assert np.allclose(norm2(B*eigv[:,i])/norm2(eigv[:,i]),eigs[i])
        x,info=spla.cg(Alo,eigv[:,i],M=mbd,maxiter=2,tol=1.e-10)
        print info
        #assert checking_output(info)


def test_eigenvalue_routine_for_symmetric_matrix():
    """
    test the routine from ARPACK but by considering hermitian matrix.

    """
    pol=2
    nt,npix,nb=400,60,1
    blocksize=nt/nb
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)
    N=BlockLO(blocksize,t,offdiag=True)
    processd =  ProcessTimeSamples(pairs,npix,pol=pol ,phi=phi)
    npix=   processd.get_new_pixel[0]
    P   =   SparseLO(npix,nt,pairs,pol=pol,angle_processed=processd)
    x0  =   np.zeros(pol*npix)
    Mbd =   BlockDiagonalPreconditionerLO(processd ,npix,pol=pol)

    x0=np.zeros(pol*npix)

    b=P.T*N*d
    A=P.T*N*P
    tol=1.e-4
    prec=(P.T*P).to_array()
    eigv1 ,Z1=spla.eigsh(A,M=P.T*P,Minv=Mbd,k=5,which='SM',ncv=12,tol=tol)
    r=Z1.shape[1]
    Az1=Z1*0.
    for i in xrange(r):
        Az1[:,i]=A*Z1[:,i]
    E1=CoarseLO(Z1,Az1,r)
    Zd1=DeflationLO(Z1)

    #Build the 2-level preconditioner
    I= lp.IdentityOperator(pol*npix)
    R1=I - A*Zd1*E1*Zd1.T
    M1=Mbd*R1 + Zd1*E1*Zd1.T
    def count_iterations(x):
        globals()['c']+=1
    for i in range(r):
        globals()['c']=0
        assert np.allclose(M1*A*Z1[:,i],Z1[:,i])
        assert norm2(R1*A*Z1[:,i])<=1.e-10
        x,info=spla.cg(M1*A,Z1[:,i],tol=tol,maxiter=2,callback=count_iterations)
        assert checking_output(info)
        assert  globals()['c']==1



filter_warnings("ignore")
test_arnoldi_algorithm()
#test_eigenvalue_routine_for_symmetric_matrix()
