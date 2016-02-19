import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def  obspix2mask(obspix,pixs,nside,fname,write=False):
    """
    From the observed pixels to a binary mask, (``mask[obspix]=1 , 0 elsewhere``)

    **Parameters**

    - ``osbpix``:{array}
        pixels observed during the scanning of the telescope.
        Already ordered in the  HEALPIX pixelization.
    - ``nside``: {int}
        Healpix parameter to define the pixelization grid of the map
    - ``fname``:{str}
        path to the fits file to store/read the map
    - ``write``:{bool}
        if ``True`` it writes onto the file, it reads from it otherwise

    **Returns**

    - mask :{array}


    """
    mask=np.zeros(hp.nside2npix(nside))
    mask[obspix[pixs]]=1.
    if write:
        hp.write_map(fname,mask)

    return mask

def reorganize_map(mapin,obspix,npix,nside,pol,fname,write=False):
    """
    From the solution map of the preconditioner to a Healpix map.
    It specially split the input array ``mapin`` which is a IQU
    for a polarization analysis in to 3 arrays ``i,q,u``.

    **Parameters**

    - ``mapin``:{array}
        solution array map (``size=npix*pol``);
    - ``obspix``:{array}
        array containing the observed pixels in the Healpix ordering;
    - ``npix``:{int}
    - ``nside``: {int}
        the same as in ``obspix2mask``;
    - ``pol``:{int}
    - ``fname``:{str}
    - ``write``:{bool}

    **Returns**

    - healpix_map:{list of arrays}
         pixelized map  with Healpix.

    """

    healpix_npix=hp.nside2npix(nside)


    if pol==3:
        healpix_map=np.zeros(healpix_npix*pol).reshape((healpix_npix,pol))
        i=mapin[np.arange(0,npix*3,3)]
        q,u=mapin[np.arange(1,npix*3,3)],mapin[np.arange(2,npix*3,3)]

        m=np.where(q!=0.)[0]
        healpix_map[obspix,0]=i
        healpix_map[obspix,1]=q
        healpix_map[obspix,2]=u
        hp_list=[healpix_map[:,0],healpix_map[:,1],healpix_map[:,2]]
    if pol==2:
        healpix_map=np.zeros(healpix_npix*pol).reshape((healpix_npix,pol))

        q,u=mapin[np.arange(0,npix*pol,2)],mapin[np.arange(1,npix*pol,pol)]

        healpix_map[obspix,0]=q
        healpix_map[obspix,1]=u
        hp_list=[healpix_map[:,0],healpix_map[:,1]]

    elif pol==1:
        healpix_map=np.zeros(healpix_npix)

        healpix_map[obspix]=mapin
        hp_list=healpix_map
    if write:
        hp.write_map(fname,hp_list)

    return hp_list

def show_map(outm,pol,patch,figname=None):
    coord_dict={'ra23':[-13.45,-32.09]}
    coords=coord_dict[patch]

    #outm[0][unseen]=hp.UNSEEN

    if pol==1:
        unseen=np.where(outm[0]==0)[0]
        outm[unseen]=hp.UNSEEN
        hp.gnomview(outm,rot=coords,xsize=600,title='I map')
        hp.graticule(dpar=5,dmer=5,local=True)
    elif pol==2:
        unseen=np.where(outm[0]==0)[0]
        outm[0][unseen]=hp.UNSEEN
        hp.gnomview(outm[0],rot=coords,xsize=600,title='Q map',sub=121)
        hp.graticule(dpar=5,dmer=5,local=True)
        outm[1][unseen]=hp.UNSEEN
        hp.gnomview(outm[1],rot=coords,xsize=600,title='U map',sub=122)
        hp.graticule(dpar=5,dmer=5,local=True)
    elif pol==3:
        unseen=np.where(outm[0]==0)[0]
        outm[0][unseen]=hp.UNSEEN
        hp.gnomview(outm[0],rot=coords,xsize=600,title='I map',sub=131)
        hp.graticule(dpar=5,dmer=5,local=True)
        outm[1][unseen]=hp.UNSEEN
        hp.gnomview(outm[1],rot=coords,xsize=600,title='Q map',sub=132)
        hp.graticule(dpar=5,dmer=5,local=True)
        outm[2][unseen]=hp.UNSEEN
        hp.gnomview(outm[2],rot=coords,xsize=600,title='U map',sub=133)
        hp.graticule(dpar=5,dmer=5,local=True)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def compare_maps(outm,inm,pol,patch,mask,figname=None):
    """
    Print on device the input map,  the one processed from datastream
    and their difference.
    """

    unseen=np.where(mask == 0. )[0]
    observ=np.where(mask != 0. )[0]

    print len(unseen)
    coord_dict={'ra23':[-13.45,-32.09]}
    coords=coord_dict[patch]
    if pol==1:
        maxval=max(inm[observ])
        minval=min(inm[observ])
        inm[unseen]=hp.UNSEEN
        outm[unseen]=hp.UNSEEN
        hp.gnomview(inm,rot=coords,xsize=600,min=minval,max=maxval,title='I input map',sub=131)
        hp.graticule(dpar=5,dmer=5,local=True)
        #hp.gnomview(outm[0],rot=coords,xsize=600,min=minval,max=maxval,title='I output',sub=132)
        hp.gnomview(outm,rot=coords,xsize=600,min=minval,max=maxval,title='I output',sub=132)

        hp.graticule(dpar=5,dmer=5,local=True)

        #diff=inm-outm[0]
        diff=inm-outm

        diff[unseen]=hp.UNSEEN
        hp.gnomview(diff,rot=coords,xsize=600,title='I diff',sub=133)
        hp.graticule(dpar=5,dmer=5,local=True)

    elif pol==3:
        maxval=max(inm[1][observ])
        minval=min(inm[1][observ])
        inm[1][unseen]=hp.UNSEEN
        outm[1][unseen]=hp.UNSEEN
        hp.gnomview(inm[1],rot=coords,xsize=600,min=minval,max=maxval,title='Q input map',sub=231)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[1],rot=coords,xsize=600,min=minval,max=maxval,title='Q output map',sub=232)
        hp.graticule(dpar=5,dmer=5,local=True)

        diff=inm[1]-outm[1]
        diff[unseen]=hp.UNSEEN
        hp.gnomview((diff),rot=coords,xsize=600,title='Q diff',sub=233)
        hp.graticule(dpar=5,dmer=5,local=True)
        maxval=max(inm[2][observ])
        minval=min(inm[2][observ])
        inm[2][unseen]=hp.UNSEEN
        outm[2][unseen]=hp.UNSEEN
        hp.gnomview(inm[2],rot=coords,xsize=600,min=minval,max=maxval,title='U input map',sub=234)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[2],rot=coords,xsize=600,min=minval,max=maxval,title='U output map',sub=235)
        hp.graticule(dpar=5,dmer=5,local=True)

        diff=inm[2]-outm[2]
        diff[unseen]=hp.UNSEEN

        hp.gnomview((diff),rot=coords,xsize=600,title='U diff',sub=236)
        hp.graticule(dpar=5,dmer=5,local=True)

    elif pol==2:
        maxval=max(inm[0][observ])
        minval=min(inm[0][observ])
        inm[0][unseen]=hp.UNSEEN
        outm[0][unseen]=hp.UNSEEN
        hp.gnomview(inm[0],rot=coords,xsize=600,min=minval,max=maxval,title='Q input map',sub=231)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[0],rot=coords,xsize=600,min=minval,max=maxval,title='Q output map',sub=232)
        hp.graticule(dpar=5,dmer=5,local=True)
        diff=inm[0]-outm[0]
        diff[unseen]=hp.UNSEEN
        hp.gnomview((diff),rot=coords,xsize=600,title='Q diff',sub=233)
        hp.graticule(dpar=5,dmer=5,local=True)
        maxval=max(inm[1][observ])
        minval=min(inm[1][observ])
        inm[1][unseen]=hp.UNSEEN
        outm[1][unseen]=hp.UNSEEN
        hp.gnomview(inm[1],rot=coords,xsize=600,min=minval,max=maxval,title='U input map',sub=234)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[1],rot=coords,xsize=600,min=minval,max=maxval,title='U output map',sub=235)
        hp.graticule(dpar=5,dmer=5,local=True)

        diff=inm[1]-outm[1]
        diff[unseen]=hp.UNSEEN

        hp.gnomview((diff),rot=coords,xsize=600,title='U diff',sub=236)
        hp.graticule(dpar=5,dmer=5,local=True)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()
    pass
