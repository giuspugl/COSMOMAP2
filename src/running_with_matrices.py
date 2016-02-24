import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from interfaces import *
from utilities import *
import scipy.sparse.linalg as spla

import time

nside=128
#runcase={'I':1,'QU':2}
runcase={'QU':2}

for id_run,pol in runcase.iteritems():
        d,t,phi,pixs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol)
        nt,npix,nb=len(d),len(hp_pixs),len(t)
        P=SparseLO(npix,nt,pixs,phi,pol=pol)
        A=P.T*P
        if pol==1:
            Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
            inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits')

        elif pol==2:
            Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
            inm=hp.read_map('data/cmb_r0.2_3.5arcmin_128.fits',field=[0,1,2])
        fname='data/map_BD_'+id_run+'_cmb_'+str(nside)+'.fits'

        start=time.clock()
        pr=profile_run()
        pr.enable()
        b=P.T*d
        pr.disable()
        output_profile(pr)
        print len(P.mask)
        end=time.clock()
        print "expected time to explicitate the matrix: %g seconds."%((end-start)*npix*pol)
        x0=np.zeros(npix*pol)

        start=time.clock()
        matr= A.to_array()
        end=time.clock()
        print "matrix A^t* A explicitation took %g seconds."%(end-start)

        start=time.clock()
        precond=Mbd.to_array()
        end=time.clock()
        print "Preconditioner explicitation took %g seconds."%(end-start)
        print matr.shape, precond.shape

        globals()['c']=0
        def count_iterations(x):
            globals()['c']+=1
        start=time.clock()
        prod=dgemm(precond,matr)
        #x,info=spla.cg(matr,b,x0=x0,M=precond,maxiter=10,callback=count_iterations)
        end=time.clock()
        maxval=prod.max()
        prod/=maxval
        imgplot=plt.imshow(prod,interpolation='nearest',vmin=prod.min(), vmax=1)
        imgplot.set_cmap('spectral')
        plt.colorbar()
        plt.show()

        """
        print "****"*15
        print " # iterations \t time to convergence [sec]\n %d \t \t%g \n "%(globals()['c'],end-start)
        print "****"*15

        assert checking_output(info)
        mask=obspix2mask(hp_pixs,pixs,nside,'data/mask_ra23.fits',write=False)
        hp_map=reorganize_map(x,hp_pixs,npix,nside,pol,fname,write=True)

        compare_maps(hp_map,inm,pol,'ra23',mask)
        """
