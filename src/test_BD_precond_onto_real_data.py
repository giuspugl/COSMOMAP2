import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla

def test_block_diagonal_precond_onto_real_data():

    """
    Test the action of the block diagonal preconditioner
    with a realistic scanning strategy.
    """

    pol=3


    d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol)

    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])
    nside=128


    print "SparseLO init"

    pr=profile_run()
    #pr.enable()
    P=SparseLO(npix,nt,pixs,phi,pol=pol)
    #pr.disable()

    A=P.T*P

    if pol==1:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
        fname='data/map_BD_i_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')

    elif pol==3:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
        fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])


    print "b=P.T*d"


    x0=np.zeros(npix*pol)
    b=P.T*d
    x=Mbd*b
    #x,info=spla.cg(A,b,x0=x0,M=Mbd,maxiter=10)
    #output_profile(pr)
    #assert checking_output(info)


    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)

    coords=[-13.45,-32.09]

    inm*=mask

    #hp_map[1]*=-1.
    #hp_map[2]*=-1.
    compare_maps(hp_map,inm,pol,coords,mask)

def subtract_ground_template(g,d,tod_size,ces_size):

    ces=tod_size/ces_size

    for i in xrange(ces):
        average=np.mean(g[i*ces_size:(i+1)*ces_size])
        print average,d[i*ces_size]
        print len(g[i*ces_size:(i+1)*ces_size])==ces_size
        d[i*ces_size:(i+1)*ces_size]-=average
    return d

def test_block_diagonal_precond_plus_noise_onto_real_data():

    """
    Test the action of the block diagonal preconditioner
    with a realistic scanning strategy.
    """

    pol=1
    d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol)
    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])
    nside=128
    print "SparseLO initialized"

    pr=profile_run()
    #pr.enable()
    #P=SparseLO(npix,nt,pixs,phi,pol=pol)
    #pr.disable()
    d=subtract_ground_template(ground,d,nt,ces_size)

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

    x=Mbd*b
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)
    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)



    x0=np.zeros(npix*pol)
    ck=0
    #x,info=spla.cg(A,b,x0=x0,M=Mbd,maxiter=10,callback=count(ck))
    #output_profile(pr)
    #assert checking_output(info)



    coords=[-13.45,-32.09]

    inm*=mask



    compare_maps(hp_map,inm,pol,coords,mask)



test_block_diagonal_precond_onto_real_data()
#test_block_diagonal_precond_plus_noise_onto_real_data()
