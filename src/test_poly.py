import numpy as np
from interfaces import *
from utilities import *
import time
from numpy.polynomial.legendre import Legendre  as poly
import pylab as pl
from scipy.special import legendre


def preprocess_ground(g):
    nbins_ground=int(max(g))+1

    negs=np.ma.masked_less(g,0)
    g[negs.mask]=-1
#    print len(g[negs.mask])-len(g[flag.mask]),"missed negative ground bins"
#    unflag=np.logical_not(negs.mask)
    groundbins=[]
    counts=np.zeros(nbins_ground)


    for i in g :
        counts[i]+=1.
        if not groundbins.__contains__(i):
            groundbins.append(i)
        else : continue
    return nbins_ground,counts,groundbins

def test_ground_filter():
    pol=1
    #filelist=['data/20120718_093931.hdf5']
    filelist=['/home/peppe/pb/mapmaker/data/20120718_050345.hdf5']

    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=100,filtersubscan=True)

    nt,npix,nb=len(d),len(hp_pixs),len(t)
#    obsp=read_obspix_from_hdf5('/home/peppe/pb/mapmaker/data/obspix.hdf5')
    #CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi,obspix2=obsp)
    CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi,ground=ground )
    npix,hp_pixs=CESs.get_new_pixel
    #flag=np.ma.masked_equal(pixs,-1)
    #ground[flag.mask]=-1
    #nbins_ground,counts,groundbins=preprocess_ground(ground)

    #print sum(counts)==nt
    #print groundbins

    P=SparseLO(npix,nt,pixs)
    Mbd=BlockDiagonalPreconditionerLO(CESs,npix,pol)
    """
    G=SparseLO(nbins_ground,nt,ground)

    print counts
    G.counts=counts

    invGtG=BlockDiagonalPreconditionerLO(G,nbins_ground)
    I= lp.IdentityOperator(nt)

    F=I -G*invGtG*G.T
    """
    F=GroundFilterLO(ground)

    v=np.ones(nt)
    m2=Mbd*P.T *d
    m=Mbd*P.T*F*d
    hp_map=reorganize_map(m,hp_pixs,npix,1024,pol)
    hp_map2=reorganize_map(m2,hp_pixs,npix,1024,pol)

    compare_maps(hp_map2,hp_map,pol,'ra23',norm='None')
    #plt.plot(d)
    #show_matrix_form(invGtG)
    #plt.plot(G*invGtG*G.T*d)
    #plt.show()
    #print F*d


def test_polynomials():
    nt=50
    npix=5
    nb=1
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)

    #x=np.arange(nt)
    polyorder=3
    #pl.plot(d,label='before filtering ')
    legendres=get_legendre_polynomials(polyorder,nt)
    print "order","< p,d > "
    for i in range(polyorder):
        #pl.plot(legendres[:,i],label='leg not norm')
        legendres[:,i]/=norm2(legendres[:,i])
        #pl.plot(legendres[:,i],label='leg norm')

        #pl.plot(q[:,i],label='qs')

        #print np.allclose(legendres[:,i],q[:,i]),legendres[:,i],q[:,i]
        #print scalprod(legendres[:,i],legendres[:,i+1]),norm2(legendres[:,i])
        projection = scalprod(legendres[:,i],d)*legendres[:,i]
        print i,projection[0],scalprod(q[:,i],d)*q[:,i][0]
        #d-=projection
        pl.plot(d-projection,label='filter w/ '+str(i)+'th order Legendre polynomial')
        pl.plot(projection,label='filter vector '+str(i)+'th order Legendre')
    print np.mean(d)
    plt.legend(loc='best')

    pl.show()
def test_poly_filtering():
    pol=1
    filelist=['/home/peppe/pb/mapmaker/data/20120718_050345.hdf5']

    #filelist=['data/20120718_093931.hdf5']
    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=None,filtersubscan=True)

    nt,npix,nb=len(d),len(hp_pixs),len(t)

    CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi)
    npix,hp_pixs=CESs.get_new_pixel
    print nt,npix,nb
    #pl.plot(d,label='before filtering ')
    P=SparseLO(npix,nt,pixs,angle_processed=CESs,pol=pol)
    Mbd=BlockDiagonalPreconditionerLO(CESs,npix,pol)
    mainm=Mbd*P.T *d
    mapp=reorganize_map(mainm,hp_pixs,npix,1024,pol)
    F=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs,poly_order=0)

    F1=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs,poly_order=1)
    F2=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs,poly_order=2)

    F3=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs,poly_order=3)
    s=time.clock()
    #fd1=F1*d
    e=time.clock()
    print "poly",e-s
    s=time.clock()
    fd= F*d
    e=time.clock()
    print "mean",e-s

    v=np.ones(nt)
    m1=reorganize_map(Mbd*P.T*F1*d, hp_pixs,npix,1024,pol )
    compare_maps(mapp,m1,pol,'ra23',norm='None')
    m=reorganize_map(Mbd*P.T*F*d, hp_pixs,npix,1024,pol )
    mapp=reorganize_map(mainm,hp_pixs,npix,1024,pol)

    compare_maps(mapp,m,pol,'ra23',norm='None')
    m2=reorganize_map(Mbd*P.T*F2*d, hp_pixs,npix,1024,pol )
    mapp=reorganize_map(mainm,hp_pixs,npix,1024,pol)

    compare_maps(mapp,m2,pol,'ra23',norm='None')
    m3=reorganize_map(Mbd*P.T*F3*d, hp_pixs,npix,1024,pol )
    mapp=reorganize_map(mainm,hp_pixs,npix,1024,pol)

    compare_maps(mapp,m2,pol,'ra23',norm='None')
    #pl.plot(fd,label='after filtering ')
    #pl.plot(fd1,label='after filtering 1')
    #plt.legend(loc='best')#

    #pl.show()
#test_poly_filtering()
test_ground_filter()
