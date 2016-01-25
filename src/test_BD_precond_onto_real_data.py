import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla

def test_block_diagonal_precond_onto_real_data():
    """
    Test the action of the block diagonal preconditioner, defined as
    :math: `M_{BD}=(A^T A)^{-1}`
    with a realistic scanning strategy.
    """
    #runcase={'IQU':3}
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol)
        nt,npix,nb=len(d),len(hp_pixs),len(t)
        print nt,npix,nb,len(hp_pixs[pixs])
        nside=128
        pr=profile_run()

        P=SparseLO(npix,nt,pixs,phi,pol=pol)
        A=P.T*P
        x0=np.zeros(npix*pol)
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
        if pol==1:
            fname='data/map_BD_i_cmb_'+str(nside)+'.fits'
            inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')
        elif pol==2:
            inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[1,2])
            fname='data/map_BD_qu_cmb_'+str(nside)+'.fits'
        elif pol==3:
            fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'
            inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])


        b=P.T*d



        globals()['c']=0
        def count_iterations(x):
            globals()['c']+=1

        pr.enable()
        #x=Mbd*b
        x,info=spla.cg(A,b,x0=x0,M=Mbd,tol=1.e-3,maxiter=10,callback=count_iterations)
        pr.disable()
        output_profile(pr)
        #checking_output(info)
        print "After  %d iteration. "%(globals()['c'])
        #assert checking_output(info) and globals()['c']==1


        hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)
        mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)


        #compare_maps(hp_map,inm,pol,'ra23',mask)



def test_block_diagonal_precond_plus_noise_onto_real_data():

    """
    Test the action of the block diagonal preconditioner, defined as
    :math: `M_{BD}=(A^T diag(N^{-1}) A)^{-1}`
    with a realistic scanning strategy.
    """
    pol=1
    d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol)
    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb,len(hp_pixs[pixs])
    nside=128
    pr=profile_run()
    N=BlockLO(nt/nb,t)
    P=SparseLO(npix,nt,pixs,phi,pol=pol,w=N.diag )
    A=P.T*N*P
    if pol==1:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
        fname='data/map_BD_i_cmb_'+str(nside)+'.fits'
        print "reading input map"
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')
        solver=spla.cg
    elif pol==3:
        Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
        fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'
        inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])
        print "reading input map"
        solver=spla.gmres
    b=P.T*N*d
    x0=np.zeros(npix*pol)

    x=Mbd*b
    globals()['c']=0
    def count_iterations(x):
        globals()['c']+=1


    pr.enable()
    #x,info=solver(A,b,x0=x0,M=Mbd,maxiter=10,callback=count_iterations)
    pr.disable()
    #output_profile(pr)

    print "After %d iteration. "%(globals()['c'])
    #assert checking_output(info) and globals()['c']==1

    #GRAFIC TOOLS
    mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)
    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)

    compare_maps(hp_map,inm,pol,'ra23',mask)


test_block_diagonal_precond_onto_real_data()
#
#test_block_diagonal_precond_plus_noise_onto_real_data()

def compute_hitmaps():
        x0+=1.
        hitmap= reorganize_map(A*x0,hp_pixs,npix,nside,pol,'data/hitmap_ra23_1day.fits')
        zeros=np.where(hitmap[0] == 0. )[0]

        hitmap[0][zeros]=hp.UNSEEN
        hp.gnomview(hitmap[0],rot=[-13.45,-32.09],xsize=600,title='hits',min=0)
        x0=np.array([0,0,1]*npix)
        hitmap= reorganize_map(A*x0,hp_pixs,npix,nside,pol,'data/hitmap_ra23_1day.fits')

        zeros=np.where(hitmap[0] == 0. )[0]
        hitmap[0][zeros]=hp.UNSEEN
        hitmap[1][zeros]=hp.UNSEEN
        hitmap[2][zeros]=hp.UNSEEN

        #hp.gnomview(hitmap[0],rot=[-13.45,-32.09],xsize=600,title='hits',min=0,sub=131)
        #hp.gnomview(hitmap[1],rot=[-13.45,-32.09],xsize=600,title='hits',sub=132)
        hp.gnomview(hitmap[2],rot=[-13.45,-32.09],xsize=600,title='hits',sub=133)
