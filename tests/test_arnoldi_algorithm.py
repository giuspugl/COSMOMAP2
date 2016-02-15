from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla
import numpy as np
def vecT_vec(size):
    v=np.random.random(size)
    v/=norm2(v)
    return  np.outer(v,v.T)

def test_arnoldi_algorithm():
    size=1000
    diag=np.logspace(-5,0,num=size)
    #print diag
    D=np.diag(diag)
    prec=np.diag(1/diag)

    A=D

    val=[min(diag),max(diag),diag[size/3],diag[-size/3],diag[size/2]]
    for i in val:
        vTv=vecT_vec(size)

        A+=i*vTv
    B=spla.aslinearoperator(dgemm(prec,A))
    b=prec.dot(np.random.random(size))
    Alo=spla.aslinearoperator(A)
    mbd=spla.aslinearoperator(prec)
    x0=np.ones(size)

    #eigs1,eigv1=spla.eigs(A,M=mbd,Minv=D,k=len(val),which='SM',ncv=24,tol=1.e-3)
    #print eigs1
    eigs,eigv=spla.eigs(B,k=len(val),which='SM',ncv=24,tol=1.e-3)

    """
    print "***"*30,"WRITING"
    write_ritz_eigenvectors_to_hdf5(eigv,'data/ritz_eigenvectors_test.hdf5')
    print "***"*30,"READING"

    Z,r=read_ritz_eigenvectors_from_hdf5('data/ritz_eigenvectors_test.hdf5',size)
    """
    r=len(eigs)
    for i in range(r):
        assert np.allclose(norm2(B*eigv[:,i])/norm2(eigv[:,i]),eigs[i])
        x,info=spla.cg(B,eigv[:,i],maxiter=3,tol=1.e-10)
        assert checking_output(info)




test_arnoldi_algorithm()
