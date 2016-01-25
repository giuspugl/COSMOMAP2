import numpy as np
import h5py as h5
from  utilities_functions import *


def read_from_data(filename,pol):
    """
    Read a hdf5 file preprocessed by the AnalysisBackend
    of the Polarbear Collaboration.

    **Parameters**

    - ``filename``:{str}
        path to the hdf5 file
    - ``pol``:{int}

      - ``1``: read data for temperature only data;

      - ``3``: read  for polarization data;


    """

    f=h5.File(filename,"r")
    hp_pixs=f['obspix'][...]
    n_bolo_pair=f['n_bolo_pair'][...]
    n_ces=f['n_sample_ces'][...]
    print "Bolo Pairs: %d \t CES: %d "%(n_bolo_pair,n_ces)

    pixs_pair=[]
    polang_pair=[]
    d_pair=[]
    weight_pair=[]
    ground_pair=[]
    #for i in range(n_bolo_pair):
    for i in range(10):
        group=f['bolo_pair_'+str(i)]
        pixs_pair.append(group['pixel'][...])
        polang_pair.append(group['pol_angle'][...])
        ground_pair.append(group['ground'][...])
        if pol== 1:
            d_pair.append(group['sum'][...])
            weight_pair.append(group['weight_sum'][...])
        elif pol==3 or pol==2:
            d_pair.append(group['dif'][...])
            weight_pair.append(group['weight_dif'][...])
    f.close()

    d=np.concatenate(d_pair)
    weight=np.array(weight_pair)
    polang=np.concatenate(polang_pair)
    pixs=np.concatenate(pixs_pair)
    ground=np.concatenate(ground_pair)

    return d,weight,polang,pixs,hp_pixs,ground,n_ces


def read_from_data_with_subscan_resize(filename,pol):
    """
    Read a hdf5 file preprocessed by the AnalysisBackend
    of the Polarbear Collaboration.

    **Parameters**

    - ``filename``:{str}
        path to the hdf5 file
    - ``pol``:{int}

      - ``1``: read data for temperature only data;

      - ``3``: read  for polarization data;


    """

    f=h5.File(filename,"r")
    hp_pixs=f['obspix'][...]
    n_bolo_pair=f['n_bolo_pair'][...]
    n_ces=f['n_sample_ces'][...]
    subscan=[f['subscans/n_sample'][...],f['subscans/t_start'][...]]
    print "Bolo Pairs: %d \t CES: %d "%(n_bolo_pair,n_ces)
    pixs_pair=[]
    polang_pair=[]
    d_pair=[]
    weight_pair=[]
    ground_pair=[]
    #for i in range(n_bolo_pair):
    for i in range(5):
        group=f['bolo_pair_'+str(i)]
        pixs_pair.append(subscan_resize(group['pixel'][...],subscan))
        polang_pair.append(subscan_resize(group['pol_angle'][...],subscan))
        ground_pair.append(subscan_resize(group['ground'][...],subscan))
        if pol== 1:
            d_pair.append(subscan_resize(group['sum'][...],subscan))
            weight_pair.append(group['weight_sum'][...])
        elif pol==3:
            d_pair.append(subscan_resize(group['dif'][...],subscan))
            weight_pair.append(group['weight_dif'][...])
    f.close()

    d=np.concatenate(d_pair)
    weight=np.array(weight_pair)
    polang=np.concatenate(polang_pair)
    pixs=np.concatenate(pixs_pair)
    ground=np.concatenate(ground_pair)

    return d,weight,polang,pixs,hp_pixs,ground,n_ces,subscan[0]


def write_ritz_eigenvectors_to_hdf5(z,filename):

    datatype=z[0][0].dtype
    if datatype == 'complex128':
        dt=h5py.special_dtype(vlen=datatype)
    else:
        dt = h5.h5t.IEEE_F64BE
    size_eigenvectors,n_eigenvals=len(z[0]),len(z)
    f=h5.File(filename,"w")
    eigenvect_group=f.create_group("Ritz_eigenvectors")
    eigenvect_group.create_dataset('n_eigenvectors',np.shape(n_eigenvals),\
                                    dtype=h5.h5t.STD_I32BE,data=n_eigenvals)
    i=0
    for tmp in z:
        eig=eigenvect_group.create_dataset('Eigenvector_'+str(i),(size_eigenvectors,),\
                                            dtype=dt)
        eig[...]=tmp
        i+=1
    return

def read_ritz_eigenvectors_from_hdf5(filename,npix):

    f=h5.File(filename,"r")
    n_eigenvals=f["Ritz_eigenvectors/n_eigenvectors"][...]
    z=np.zeros(n_eigenvals*npix).reshape(npix,n_eigenvals)
    tmp=[]
    for i in xrange(n_eigenvals):
        tmp=f["Ritz_eigenvectors/Eigenvector_"+str(i)][...]
        z[:,i]=tmp
    return z


def write_to_hdf5(filename,obs_pixels,noise_values,d,phi=None):
    """
    Write onto hdf5 file whose datasets are created by the routine
    ``utilities_functions.system_setup``

    """

    f=h5.File(filename,"w")
    group=f.create_group("bolo_pair")

    pixs=group.create_dataset('pixel',np.shape(obs_pixels), dtype=h5.h5t.STD_I32BE)
    weight=group.create_dataset('weight',np.shape(noise_values),dtype=h5.h5t.IEEE_F64BE)
    det=group.create_dataset('sum',np.shape(d),dtype=h5.h5t.IEEE_F64BE)
    if phi is not None:
        polang=group.create_dataset('pol_angle',np.shape(phi), dtype=h5.h5t.IEEE_F64BE)
        polang[...]=phi

    pixs[...]=obs_pixels
    weight[...]=noise_values
    det[...]=d


    f.close()
    pass

def show_matrix_form(A):
    import matplotlib.pyplot as plt
    matr=A.to_array()
    #print matr
    maxval=matr.max()
    matr/=maxval
    imgplot=plt.imshow(matr,interpolation='nearest',vmin=matr.min(), vmax=1)
    imgplot.set_cmap('spectral')
    plt.colorbar()
    plt.show()
    pass

def read_from_hdf5(filename):
    """
    Read from a hdf5 file whose datasets are created by the routine
    ``utilities_functions.system_setup``
    """
    f=h5.File(filename,"r")

    obs_pix=f['/bolo_pair/pixel'][...]
    polang=f['/bolo_pair/pol_angle'][...]
    weight=f['/bolo_pair/weight'][...]

    det=f['bolo_pair/sum'][...]
    f.close()

    return det,obs_pix,polang,weight

def plot_histogram_eigenvalues(z):
    """
    save a plot containing an histogram  of the eigenvalues ``z``
    """

    import matplotlib.pyplot as plt
    from matplotlib import rc

    histo,edges=np.histogram(abs(z),bins=20,normed=False)
    bins=np.array([(edges[i]+edges[i+1])/2. for i in range(len(histo))])
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Eigenvalues Histogram')
    plt.xlabel(r'$\lambda_i $')
    plt.hist(histo,bins=bins,color='b', linewidth=1.5)
    plt.savefig('data/eigenvalues_histogram.png')
    pass
