import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla
import sys
import scipy.linalg as la
import time
import krypy as kp

filter_warnings("ignore")
def test_myarnoldi():

    """
    Test the action of the 2-level  preconditioner
    with a realistic scanning strategy.
    """
    nside=1024
    pol=2
    #filelist=['data/20120718_093931.hdf5','data/20131011_092136.hdf5']
    #filelist=['/home/peppe/pb/mapmaker/data/20120718_050345.hdf5']
    filelist=['data/20120718_093931.hdf5']
    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=10,filtersubscan=True)
                #read_from_data_with_subscan_resize('data/20120718_093931.hdf5',pol=pol)

    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb

    CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi)
    npix,hp_pixs=CESs.get_new_pixel

    P=SparseLO(npix,nt,pixs,angle_processed=CESs,pol=pol)
    Mbd=BlockDiagonalPreconditionerLO(CESs,npix,pol)

    F=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,\
                bolos_per_ces,pixs,poly_order=1)

    b=P.T*F*d
    A=P.T*F*P
    v,h,m=arnoldi(Mbd*A, Mbd*b, x0=np.ones(pol*npix), tol=1e-5, inner_m=pol*npix )
    print m
    H=build_hess(h,m)
    from  numpy import linalg  as la
    z,y=la.eigh(H)
    thresh=1.e-2
    Z,r=build_Z(z,y,v,thresh)
    print z[:r]
    return z[:r]



def test_arnoldi_krypy():

    """
    Test the action of the 2-level  preconditioner
    with a realistic scanning strategy.
    """
    nside=1024
    pol=1
    #filelist=['data/20120718_093931.hdf5','data/20131011_092136.hdf5']
    filelist=['/home/peppe/pb/mapmaker/data/20120718_050345.hdf5']
    #filelist=['data/20120718_093931.hdf5']
    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=10,filtersubscan=True)
                #read_from_data_with_subscan_resize('data/20120718_093931.hdf5',pol=pol)

    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb

    CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi)
    npix,hp_pixs=CESs.get_new_pixel

    P=SparseLO(npix,nt,pixs,angle_processed=CESs,pol=pol)
    Mbd=BlockDiagonalPreconditionerLO(CESs,npix,pol)

    F=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,\
                bolos_per_ces,pixs,poly_order=0)

    b=P.T*F*d
    A=P.T*F*P

    pr=profile_run()

    x0=np.ones(pol*npix)
    tol=1.e-5
    V,H,m=run_krypy_arnoldi(A,x0,Mbd,tol)
    Z,r  =find_ritz_eigenvalues(H,V)

    #write_ritz_eigenvectors_to_hdf5(Z,'data/ritz_eigenvectors_'+P.maptype+'_filter_20120718_093931.hdf5')
    #Z,r=read_ritz_eigenvectors_from_hdf5('data/ritz_eigenvectors_'+P.maptype+'_filter_20120718_093931.hdf5')
    Zd=DeflationLO(Z)
    Az=Z*0.
    for i in xrange(r):
        Az[:,i]=A*Z[:,i]

    AZd=DeflationLO(Az)

    # Build Coarse operator
    E=CoarseLO(Z,Az,r)
    #Build the 2-level preconditioner
    I= lp.IdentityOperator(pol*npix)
    R=I - AZd*E*Zd.T

    M2=Mbd*R + Zd*E*Zd.T
    print "\t MAZ=Z \t  RAZ=0 \t#ITERATIONS cg(A,Z,M2) \t w=||MAz||/||z|| \t ||z||\t (z_i , z_j)\n"

    for i in range(r):
        globals()['c']=0
        x,info=spla.cg(A,Z[:,i],M=M2,x0=x0,tol=tol,maxiter=2,callback=count_iterations)
        n_iters=globals()['c']
        print "%d \t %r \t %r \t %d \t %g \t %g\t %g "%(i,np.allclose(M2*Az[:,i],Z[:,i]),norm2(R*Az[:,i])<=1.e-10,n_iters,norm2(Mbd*A*Z[:,i])/norm2(Z[:,i]),
                                                        norm2(Z[:,i]),scalprod(Z[:,i],Z[:,i-1]))
        #hp_map=reorganize_map(Z[:,i],hp_pixs,npix,nside,pol)
        #show_map(hp_map,pol,'ra23',norm=None)



def test_M2_w_arpack():
    """
    Test the action of the 2-level  preconditioner
    with a realistic scanning strategy.
    """
    nside=128
    pol=1
    d,t,phi,pixs,hp_pixs,ground,ces_size,subscan_nsample=\
                read_from_data_with_subscan_resize('data/20120718_093931.hdf5',pol=pol)
    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])

    N=BlockLO(nt/nb,t,offdiag=False)

    pr=profile_run()

    F=FilterLO(nt,subscan_nsample)

    P=SparseLO(npix,nt,pixs,phi,pol=pol,w=N.diag)
    A=P.T*F*P

    if pol==1:
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol)
        fname='data/map_BD_i_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')

    elif pol==3:
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol)
        fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])


    x0=np.zeros(pol*npix)
    # Build deflation supspace
    tol=1.e-7
    B=Mbd*A
    b=P.T*F*d
    tstart=time.clock()
    #w,h=arnoldi(B,Mbd*b,x0=x0,tol=tol,maxiter=1,inner_m=pol*npix)
    eigs,eigv=spla.eigs(B,k=30,which='SM',ncv=64,tol=tol,maxiter=70)
    print eigv.shape
    mask=np.where(eigs<=1.e-8)[0]
    Z=np.delete(eigv,mask,axis=1)
    print Z.shape
    tend=time.clock()
    r=Z.shape[1]
    print "selected %d eigenvectors w/ eigenvalues :\n "%(r)
    print "Arnoldi algorithm took %g minutes."%((tend-tstart)/60.)
    """
    write_ritz_eigenvectors_to_hdf5(Z,'data/ritz_eigenvectors_'+P.maptype+'_filter_20120718_093931.hdf5')
    Z,r=read_ritz_eigenvectors_from_hdf5('data/ritz_eigenvectors_'+P.maptype+'_filter_20120718_093931.hdf5',pol*npix)
    """
    Zd=DeflationLO(Z)
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
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)
    for i in range(r):
        globals()['c']=0
        #hp_map=reorganize_map(Z[:,i],hp_pixs,npix,nside,pol,fname,write=False)
        #show_map(mask*hp_map,pol,'ra23')
        x,info=spla.cg(M2*A,Z[:,i],callback=count_iterations,maxiter=4)
        n_iters=globals()['c']
        assert checking_output(info) and n_iters==1
        assert (np.allclose(M2*A*Z[:,i].real,Z[:,i].real) and norm2(R*A*Z[:,i])<=1.e-10)


def count_iterations(x):
    globals()['c']+=1
    #print globals()['c']



s=time.clock()
test_arnoldi_krypy()
e=time.clock()
t2=e-s
print t2
#test_M2_w_arpack()
