import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla
import scipy.linalg as la
import time
def test_M2_precond_onto_real_data():

    """
    Test the action of the 2-level  preconditioner
    with a realistic scanning strategy.
    """
    nside=128
    pol=3
    filelist=['data/20120718_093931.hdf5','data/20131011_092136.hdf5']
    #filelist=['data/20120718_093931.hdf5']
    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=5,filtersubscan=True)
                #read_from_data_with_subscan_resize('data/20120718_093931.hdf5',pol=pol)
    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])
    N=BlockLO(nt/nb,t,offdiag=False)

    P=SparseLO(npix,nt,pixs,phi,pixel_schema=hp_pixs,pol=pol,w=N.diag)
    pr=profile_run()

    npix=P.ncols
    hp_pixs=P.obspix
    F=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs)
    b=P.T*F*d
    """

    A=P.T*F*P

    if pol==1:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
        fname='data/map_BD_i_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')

    elif pol==3:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
        fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])


    x0=np.zeros(pol*npix)
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

    AZ=[]
    for i in Zd.z:
        AZ.append(A*i)

    for i in range(r):
        assert (np.allclose(M2*AZ[i],Zd.z[i]) and norm2(R*AZ[i])<=1.e-10)

    x0=np.zeros(pol*npix)

    globals()['c']=0
    def count_iterations(x):
        globals()['c']+=1

    pr.enable()
    x,info=spla.cg(A,b,x0=x0,tol=tol,maxiter=100,M=M2,callback=count_iterations)
    pr.disable()
    print "After %d iteration. "%(globals()['c'])

    output_profile(pr)
    checking_output(info)

    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)

    compare_maps(hp_map,inm,pol,'ra23',mask)
    """
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





test_M2_precond_onto_real_data()

#test_M2_w_arpack()
