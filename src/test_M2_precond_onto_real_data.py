import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla

def test_M2_precond_onto_real_data():

    """
    Test the action of the 2-level  preconditioner
    with a realistic scanning strategy.
    """
    pol=3
    d,t,phi,pixs,hp_pixs,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol)
    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])
    nside=128
    print "SparseLO initialized"

    pr=profile_run()
    #pr.enable()
    #P=SparseLO(npix,nt,pixs,phi,pol=pol)
    #pr.disable()

    N=BlockLO(nt/nb,t,offdiag=False)
    noiseweights=N*np.ones(nt)
    P=SparseLO(npix,nt,pixs,phi,pol=pol,w=noiseweights)
    A=P.T*P

    if pol==1:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
        fname='data/map_BD_i_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')

    elif pol==3:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
        fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])

    b=P.T*N*d

    x0=np.zeros(pol*npix)
    # Build deflation supspace
    h=[]
    w=[]
    tol=1.e-2
    w,h=arnoldi(Mbd*A,b,x0=x0,tol=tol,maxiter=1,inner_m=pol*npix)
    m=len(w)
    
    """
    H=build_hess(h,m)
    z,y=la.eig(H,check_finite=False)
    eps=.1*abs(max(z))
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
    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)

    coords=[-13.45,-32.09]

    inm*=mask

    #hp_map[1]*=-1.
    #hp_map[2]*=-1.

    compare_maps(hp_map,inm,pol,coords,mask)

    """



test_M2_precond_onto_real_data()
