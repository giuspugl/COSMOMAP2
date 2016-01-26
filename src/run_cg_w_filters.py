import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla
import argparse
import time



globals()['c']=0
def count_iterations(x):
    globals()['c']+=1


def main(args):
    runcase={'I':1,'QU':2,'IQU':3}
    pol=runcase[args.idrun]
    fieldmap=range(pol)
    inm=hp.read_map(args.inmap,field=fieldmap)

    nside=args.nside
    d,t,phi,pixs,hp_pixs,ground,ces_size,subscan_nsample=read_from_data_with_subscan_resize(args.infile,pol=pol)
    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])
    N=BlockLO(nt/nb,t,offdiag=False)
    F=FilterLO(nt,subscan_nsample)
    P=SparseLO(npix,nt,pixs,phi,pol=pol,w=N.diag)
    x0=np.zeros(pol*npix)
    A=P.T*F*P
    Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
    b=P.T*F*d
    if args.precond=='bd':
        start=time.clock()
        x,info=spla.cg(A,b,x0=x0,M=Mbd,tol=1.e-3,maxiter=100,callback=count_iterations)
        end=time.clock()

    else:
        # Build deflation supspace
        h=[]
        w=[]
        tol=1.e-7
        B=Mbd*A

        tstart=time.clock()
        w,h=arnoldi(B,Mbd*b,x0=x0,tol=tol,maxiter=1,inner_m=pol*npix)
        tend=time.clock()
        print "Arnoldi algorithm took %g minutes."%((tend-tstart)/60.)

        m=len(w)

        H=build_hess(h,m)
        z,y=la.eig(H,check_finite=False)

        # smaller eigenvalues <30% of the energy
        total_energy=np.sqrt(sum(abs(z)**2))
        #eps=.4*abs(max(z))
        eps= .2 * total_energy

        Z,r= build_Z(z,y, w, eps)

        Zd=DeflationLO(Z)
        write_ritz_eigenvectors_to_hdf5(Zd.z,'data/ritz_eigenvectors_'+P.maptype+'_filter_20120718_093931.hdf5')
        #Z=read_ritz_eigenvectors_from_hdf5('data/ritz_eigenvectors_filter_20120718_093931.hdf5',pol*npix)
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]

        AZd=DeflationLO(Az)
        # Build Coarse operator
        E=CoarseLO(Z,Az,r)
        #Build the 2-level preconditioner
        I= lp.IdentityOperator(pol*npix)

        #R=I - A*Zd*E*Zd.T
        R=I - AZd*E*Zd.T

        M2=Mbd*R + Zd*E*Zd.T
        start=time.clock()
        x,info=spla.cg(A,b,x0=x0,M=M2,tol=1.e-3,maxiter=100,callback=count_iterations)
        end=time.clock()

    print "After  %d iteration. "%(globals()['c'])

    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,args.outmap,write=True)
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Read files and flags')
	parser.add_argument('-i','--inputfile',dest='infile',action='store',type=str,help='path of the filename')
	parser.add_argument('-f','--inputmap',dest='inmap',action='store',type=str,help='path of the input map')
	parser.add_argument('--runcase',dest='idrun',action='store',type=str,help='id of the runcase')
	parser.add_argument('--nside',dest='nside',action='store',type=int,help='healpix gridding size')
	parser.add_argument('-o','--outputmap',dest='outmap',action='store',type=str,help='path of the output map')
	parser.add_argument('--preconditioner',dest='precond',action='store',type=str,help='either bd or 2l')

	args = parser.parse_args()

	main(args)
