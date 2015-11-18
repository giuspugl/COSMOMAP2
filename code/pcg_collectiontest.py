import argparse
import scipy.sparse.linalg as spla
import numpy as np
import argparse
import linop.linop as lp
import linop.blkop as blk
import random as rd
import sys

sys.path.append('/home/peppe/pb/pcg_mapmaking/lib/')

from linearoperators import *
from utilities_functions import *
from deflationlib import *

def test0(args):
    """
    test the matrix vector multiplication of A and A^T.
    """
    print "\n///////////////////////////\n"
    print "\t TEST 0"
    print "\n///////////////////////////\n"

    x=np.arange(args.npix)

    pairs=[]
    for i in xrange(args.nt):
        pairs.append((i,rd.randint(0 ,args.npix-1)))

    P=SparseLO(args.npix,args.nt,pairs)

    #print "x\n",x
    #print "(i,j)\n",pairs
    y=P*x
    y2=[x[i[1]] for i in pairs]

    #print "P*x\n",y
    #print "y2\n",y
    #print "P.T*P*x\n",P.T*P*x
    if np.allclose(y,y2):
        print "TEST PASSED!"
    else:
        print "TEST FAILED. "


def test1(args):
    """
    test matrix vector multiplication of A^T N^{-1} A
    """
    print "\n///////////////////////////\n"
    print "\t TEST 1"
    print "\n///////////////////////////\n"


    x=np.arange(args.npix)

    pairs=[]
    for i in xrange(args.nt):
        pairs.append((i,rd.randint(0 ,args.npix-1)))

    P=SparseLO(args.npix,args.nt,pairs)
    #construct the block diagonal operator
    t=np.random.random(args.nb)

    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=False )

    y=P*x
    w=N*y
    z=P.T*w

    z2=P.T*N*P*x

    if np.allclose(z2,z):
        print "TEST PASSED!"
    else:
        print "TEST FAILED."


def test2(args):
    """
    Verify whether the built Linear Operators interfaces
    with scipy.sparse routines: CG and GMRES.

    """
    print "\n///////////////////////////\n"
    print "\t TEST 2"
    print "\n///////////////////////////\n"


    d=np.arange(args.nt)

    pairs=pairs_gen(args.nt,args.npix)
    P=SparseLO(args.npix,args.nt,pairs)

    #construct the block diagonal operator
    t=np.random.random(args.nb)
    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=False )

    x0=np.ones(args.npix)

    b=P.T*N*d

    #print 'd \n',d
    #print 'b\n',b
    #print 'x0\n',x0
    #print 'A*x0\n', P.T*N*P*x0
    print '****************************\n \t Scipy CG \t \n****************************\n'

    x,info=spla.cg(P.T*N*P,b,x0=x0, maxiter=args.npix)
    print 'SOLUTION\n', x

    checking_output(info)


    print '****************************\n \t Scipy GMRES \t \n****************************\n'
    y,info=spla.gmres(P.T*N*P,b,x0=x0, maxiter=args.npix )

    checking_output(info)
    print 'SOLUTION\n', y
    if np.mean(x-y)<=args.tol :
        print "TEST PASSED!"
    else:
        print "TEST FAILED."
        print "< x1 - x2> =%g > %g"%(np.mean(x-y),args.tol)


def test3(args):
    """
    Build the block diagonal preconditioner and check
    its action when applied onto a vector

    """
    print "\n///////////////////////////\n"
    print "\tTEST 3"
    print "\n///////////////////////////\n"

    d=np.arange(args.nt)

    pairs=pairs_gen(args.nt,args.npix)
    P=SparseLO(args.npix,args.nt,pairs)

    #construct the block diagonal operator
    t=np.random.random(args.nb)
    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=False )

    x0=np.ones(args.npix)

    b=P.T*N*d

    M_bd=InverseLO(P.T*N*P,method=spla.cg)
    vec=M_bd*b

    checking_output(M_bd.converged)
    print "x=M*b\n", vec

    print "x=CG(A,b)"
    y,info=spla.cg(P.T*N*P,b)
    checking_output(info)
    print 'SOLUTION \n', y


    if np.allclose(vec,y):
        print "TEST PASSED!"
    else:
        print "TEST FAILED."

import scipy.linalg as la
def test4(args):
    """
    Build and test the deflation subspace matrix Z checking
    whether its columns are linearly independent.
    """

    print "\n///////////////////////////\n"
    print "\tTEST 4"
    print "\n///////////////////////////\n"

    d=np.random.random(args.nt)
    pairs=pairs_gen(args.nt,args.npix)
    P=SparseLO(args.npix,args.nt,pairs)
    #construct the block diagonal operator
    bandsize=2
    t, diag=noise_val(args.nb,bandsize)

    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=True)
    diagN=BlockLO(blocksize,diag,offdiag=False)
    x0=np.zeros(args.npix)

    b=P.T*N*d
    A=P.T*N*P
    M=InverseLO(P.T*diagN*P,method=spla.cg)

    # Build deflation supspace
    h=[]
    w=[]
    w,h=arnoldi(M*A,b,x0=x0,tol=args.tol,maxiter=1,inner_m=args.npix)

    m=len(w)

    H=build_hess(h,m)
    z,y=la.eig(H,check_finite=False)
    Z,r= build_Z(z,y, w, args.eps)
    Z=np.matrix(Z).reshape((args.npix,r))

    rank= np.linalg.matrix_rank(Z)
    prod=Z.T.dot(Z)

    determ=la.det(prod)
    if (determ!=0 and rank== r):
        print "The columns of Z are linearly independent. det|Z^T*Z|=%.1g"%determ
        print "TEST PASSED!"
    else:
        print "The columns of Z are NOT linearly independent!"
        print "TEST FAILED."


def test5(args):
    """
    Build and test the coarse operator E.
    """

    print "\n///////////////////////////\n"
    print "\tTEST 5"
    print "\n///////////////////////////\n"

    d=np.random.random(args.nt)
    pairs=pairs_gen(args.nt,args.npix)
    P=SparseLO(args.npix,args.nt,pairs)

    #construct the block diagonal operator
    bandsize=2
    t, diag=noise_val(args.nb,bandsize)

    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=True)
    diagN=BlockLO(blocksize,diag,offdiag=False)
    x0=np.zeros(args.npix)

    b=P.T*N*d
    A=P.T*N*P
    M=InverseLO(P.T*diagN*P,method=spla.cg)

    # Build deflation supspace
    h=[]
    w=[]
    w,h=arnoldi(M*A,b,x0=x0,tol=args.tol,maxiter=1,inner_m=args.npix)
    m=len(w)
    H=build_hess(h,m)
    z,y=la.eig(H,check_finite=False)
    Z,r= build_Z(z,y, w, args.eps)


    # Build Coarse operator
    AZ=[]
    for i in Z:
        AZ.append(A*i)

    E=CoarseLO(Z,AZ,r)
    x=np.ones(r)
    y=E*x
    y2= la.solve(E.L.dot(E.U),x)
    if np.allclose(y2,y):
        print "TEST PASSED!"
    else:
        print "TEST FAILED."

def test6(args):
    """
    Build and test the expected algebraic properties
    of  the M2 level preconditioner.
    """

    print "\n///////////////////////////\n"
    print "\tTEST 6"
    print "\n///////////////////////////\n"

    d=np.random.random(args.nt)
    pairs=pairs_gen(args.nt,args.npix)
    P=SparseLO(args.npix,args.nt,pairs)

    #construct the block diagonal operator
    bandsize=2
    t, diag=noise_val(args.nb,bandsize)

    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=True)
    diagN=BlockLO(blocksize,diag,offdiag=False)
    x0=np.zeros(args.npix)

    b=P.T*N*d
    A=P.T*N*P
    M=InverseLO(P.T*diagN*P,method=spla.cg)

    # Build deflation supspace
    h=[]
    w=[]
    w,h=arnoldi(M*A,b,x0=x0,tol=args.tol,maxiter=1,inner_m=args.npix)
    m=len(w)
    H=build_hess(h,m)
    z,y=la.eig(H,check_finite=False)
    Z,r= build_Z(z,y, w, args.eps)

    Zd=DeflationLO(Z)

    # Build Coarse operator
    AZ=[]
    for i in Z:
        AZ.append(A*i)

    E=CoarseLO(Z,AZ,r)

    #Build the 2-level preconditioner

    I= lp.IdentityOperator(args.npix)

    R=I - A*Zd*E*Zd.T
    M2=M*R+ Zd*E*Zd.T

    for i in range(r):
        if (np.allclose(M2*AZ[i],Z[i]) and norm2(R*AZ[i])<=1.e-10):
            print "TEST PASSED!"
        else:
            print "TEST FAILED."
            raise RuntimeError("The preconditioner does not look to be well implemented.\n \
                                M2*Az==z %b and RAz=%.1g"
                                %(np.allclose(M2*AZ[i],Z[i]),norm2(R*AZ[i])) )


def test7(args):
    """
    Test the solution to the linear system Ax=b via the M2 level preconditioner.
    """

    print "\n///////////////////////////\n"
    print "\tTEST 7"
    print "\n///////////////////////////\n"

    d=np.random.random(args.nt)
    pairs=pairs_gen(args.nt,args.npix)
    P=SparseLO(args.npix,args.nt,pairs)

    #construct the block diagonal operator
    bandsize=2
    t, diag=noise_val(args.nb,bandsize)

    blocksize=args.nt/args.nb
    N=BlockLO(blocksize,t,offdiag=True)
    diagN=BlockLO(blocksize,diag,offdiag=False)
    x0=np.zeros(args.npix)

    b=P.T*N*d
    A=P.T*N*P
    M=InverseLO(P.T*diagN*P,method=spla.cg)

    # Build deflation supspace
    h=[]
    w=[]
    w,h=arnoldi(M*A,b,x0=x0,tol=args.tol,maxiter=1,inner_m=args.npix)
    m=len(w)
    H=build_hess(h,m)
    z,y=la.eig(H,check_finite=False)
    Z,r= build_Z(z,y, w, args.eps)

    Zd=DeflationLO(Z)

    # Build Coarse operator
    AZ=[]
    for i in Z:
        AZ.append(A*i)

    E=CoarseLO(Z,AZ,r)

    #Build the 2-level preconditioner

    I= lp.IdentityOperator(args.npix)

    R=I - A*Zd*E*Zd.T
    M2=M*R+ Zd*E*Zd.T

    x0=np.zeros(args.npix)
    x,info=spla.gmres(A,b,x0=x0,tol=args.tol,maxiter=100,M=M2)
    checking_output(info)

def main(args):

	test0(args)
	test1(args)
	test2(args)
	test3(args)
	test4(args)
	test5(args)
	test6(args)
	test7(args)

    #from test8 import test8
    #test8(args)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='arguments to feed the script')
	parser.add_argument('-NT',dest='nt',action='store',type=int,help='size of  tod ')
	parser.add_argument('-NP',dest='npix',action='store',type=int, help='number of pixel of the map')
	parser.add_argument('-NB',dest='nb',action='store',type=int, help='number of toeplitz blocks ')
	parser.add_argument('--tolerance',dest='tol',action='store',type=float,help='tolerance in computing Ritz eigenvalues ')
	parser.add_argument('--eigthresh',dest='eps',action='store',type=float,help='threshold for select Ritz eigenvectors')
	parser.add_argument('-pol','--polarization',dest='pol',action='store',type=int ,help='polarization maps.values=[0],2,3 ',default =0)
	args = parser.parse_args()
	main(args)
