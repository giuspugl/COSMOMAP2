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

    pol=1
    d,t,phi,pixs,hp_pixs=read_from_data('data/20120718_093931.hdf5',pol=pol)

    nt,npix,nb=len(d),len(hp_pixs),len(t)
    print nt,npix,nb
    nside=128


    print "SparseLO init"

    P=SparseLO(npix,nt,pixs,phi,pol=pol)

    A=P.T*P

    if pol==1:
        Mbd=BlockPrec(P.counts,P.mask,npix,pol)
        fname='data/map_BD_i_cmb_'+str(nside)+'.fits'

    if pol==3:
        Mbd=BlockPrec(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
        fname='data/map_BD_iqu_cmb_'+str(nside)+'.fits'

    print "b=P.T*d"

    b=P.T*d

    x0=np.zeros(npix*pol)
    pr=profile_run()
    pr.enable()
    ck=0
    x,info=spla.cg(A,b,x0=x0,M=Mbd,maxiter=10,callback=count(ck))
    pr.disable()
    output_profile(pr)

    assert checking_output(info)


    hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=False)
    inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')
    mask=obspix2mask(hp_pixs[pixs],nside,'data/mask_ra23.fits',write=False)
    inm*=mask

    #compare_maps(hp_map,inm,pol,coords)




test_block_diagonal_precond_onto_real_data()
