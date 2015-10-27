import numpy as np
import h5py as h5



def read_from_data(filename,pol):
    """
    Read a hdf5 file preprocessed by the AnalysisBackend
    of the Polarbear Collaboration.

    **Parameters**
    - filename:{str}
        path to the hdf5 file
    - pol :{int}
        - ``1``
        read data for temperature only data;

        - ``3``
        read GQU for polarization only data;
    """

    f=h5.File(filename,"r")
    hp_pixs=f['obspix'][...]
    n_bolo_pair=f['n_bolo_pair'][...]
    n_ces=f['n_sample_ces'][...]
    print "Bolo Pairs: %d \t CES: %d "%(n_bolo_pair,n_ces)

    group=f['bolo_pair_0']
    pixs=group['pixel'][...]
    polang=group['pol_angle'][...]
    if pol== 1:
        d=group['sum'][...]
        weight=group['weight_sum'][...]
    elif pol==3:
        d=group['dif'][...]
        weight=group['weight_dif'][...]
        polang+=np.pi/2.

    #for i in range(n_bolo_pair):
    for i in range(1,5):
        group=f['bolo_pair_'+str(i)]

        pixs_pair=group['pixel'][...]
        pixs=np.append(pixs,pixs_pair)
        polang_pair=group['pol_angle'][...]
        if pol== 1:
            d_pair=group['sum'][...]
            d=np.append(d,d_pair)
            weight_pair=group['weight_sum'][...]
            weight=np.append(weight,weight_pair)
        elif pol==3:
            d_pair=group['dif'][...]
            d=np.append(d,d_pair)
            weight_pair=group['weight_dif'][...]
            weight=np.append(weight,weight_pair)

            polang_pair+=np.pi/2.
        polang=np.append(polang,polang_pair)

    f.close()

    return d,weight,polang,pixs,hp_pixs


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
    if phi!=None:
        polang=group.create_dataset('pol_angle',np.shape(phi), dtype=h5.h5t.IEEE_F64BE)
        polang[...]=phi

    pixs[...]=obs_pixels
    weight[...]=noise_values
    det[...]=d


    f.close()
def read_from_hdf5(filename):
    """
    Read from a hdf5 file whose datasets are created by the routine
    ``utilities_functions.system_setup``
    """
    f=h5.File(filename,"r")

    obs_pix=f['/bolo_pair/pixel'][...]
    polang=f['/bolo_pair/pol_angle'][...]
    weight=[i for i in f['/bolo_pair/weight'][...]]
    diag=[i[0] for i in weight]
    det=f['bolo_pair/sum'][...]
    f.close()

    nt=len(obs_pix)
    pairs=[(i,obs_pix[i]) for i in xrange(nt)]

    return pairs,polang,weight,diag,det
