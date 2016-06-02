import numpy as np
from scipy import weave
from scipy.weave import inline
from  utilities_functions import *



class ProcessTimeSamples(object):
    """
    This class  precomputes quantities needed during the analysis once the input file have been read.
    During its initialization,  a private member function :func:`initializeweights`
    is called to precompute arrays needed for the explicit implementation of :class:`BlockDiagonalPreconditionerLO`
     and :class:`BlockDiagonalLO`.
    Moreover it masks all the unobserved or pathological pixels which won't be taken into account,
    via the functions  :func:`repixelization`  and   :func:`flagging_samples`.
    .. note::

        This the reason why the value ``npix`` has to be updated after the removal
         of the pathological pixels.

    **Parameters**

    - ``npix`` : {int}
        total number of pixels that could be observed;
    - ``pixs`` : {array}
        list of pixels observed in the time domain;
    - ``pol`` : {int,[*default* `pol=1`]}
        process an intensity only (``pol=1``), polarization only ``pol=2``
        and intensity+polarization map (``pol=3``);
    - ``phi``: {array, [*default* `None`]}
        array with polarization angles (needed if ``pol=3,2``);
    - ``w``: {array, [*default* `None`]}
        array with noise weights , :math:`w_t= N^{-1} _{tt}`, computed by
        :func:`BlockLO.build_blocks`.   If it is  not set :func:`ProcessTimeSamples.initializeweights`
        assumes it to be a :func:`numpy.ones` array;
    - ``obspix``:{array}
        Map from the internal pixelization to an external one, i.e. HEALPIX, it has to be modified when
        pathological pixels are not taken into account;
        Default is :func:`numpy.arange(npix)`;
    - ``threshold_cond``: {float}
        set the condition number threshold to mask bad conditioned pixels (it's used in polarization cases).
        Default is set to 1.e3.


    """
    def __init__(self, pixs,npix,obspix=None,pol=1 ,phi=None,w=None,threshold_cond=1.e3,obspix2=None):
        self.pixs= pixs
        self.oldnpix=npix
        self.nsamples=len(pixs)
        self.pol=pol
        if w is None:
            w=np.ones(self.nsamples)
        if obspix  is None:
            obspix =np.arange(self.nsamples)
        self.obspix=obspix
        if obspix2 is None:
            self.threshold=threshold_cond
            self.initializeweights(phi,w)
            self.repixelization()
            self.flagging_samples()
        else:
            self.SetObspix(obspix2)
            self.flagging_samples()
            self.compute_arrays(phi,w)

    @property
    def get_new_pixel(self):
        return self.__new_npix,self.obspix

    def SetObspix(self,new_obspix):
        n_new_pix=0
        n_removed_pix=0
        self.old2new=np.zeros(self.oldnpix,dtype=int)
        for jpix in xrange(self.oldnpix):
            if self.obspix[jpix] in new_obspix:
                self.old2new[jpix]=n_new_pix
                n_new_pix+=1
            else:
                self.old2new[jpix]=-1
                n_removed_pix+=1

        c=bash_colors()
        print c.header("___"*30)
        print c.blue("%d pixels excluded\nRepixelization  w/ %d pixels."%(n_removed_pix,n_new_pix))
        print c.header("___"*30)
        self.obspix=new_obspix
        self.__new_npix=n_new_pix

    def compute_arrays(self,w,phi):
        npix=self.__new_npix
        N=self.nsamples
        pixs=self.pixs
        if self.pol==1:
            self.counts=np.zeros(npix)
            counts=self.counts
            includes=r"""
            #include <stdio.h>
            #include <omp.h>
            #include <math.h>
            """
            code = """
            int i,pixel;
            for ( i=0;i<N;++i){
                pixel=pixs(i);
                if (pixel == -1) continue;
                counts(pixel)+=w(i);
                }
            """
            inline(code,['pixs','w','counts','N'],verbose=1,
            extra_compile_args=['-march=native  -O3  -fopenmp ' ],
                		    support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)
        else:
            self.cos=np.cos(2.*phi)
            self.sin=np.sin(2.*phi)
            self.cos2=np.zeros(npix)
            self.sin2=np.zeros(npix)
            self.sincos=np.zeros(npix)
            cos,sin =   self.cos,self.sin
            cos2,sin2,sincos=   self.cos2,self.sin2,self.sincos
            if self.pol==2:
                includes=r"""
                #include <stdio.h>
                #include <omp.h>
                #include <math.h>
                """
                code = """
                int i,pixel;
                for ( i=0;i<N;++i){
                    pixel=pixs(i);
                    if (pixel == -1) continue;
                    cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                    sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                    sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                    }
                """
                inline(code,['pixs','w','cos','sin','cos2','sin2','sincos','N'],verbose=1,
                extra_compile_args=['-march=native  -O3  -fopenmp ' ],
                support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)
            elif self.pol==3:
                self.counts=np.zeros(npix)
                self.cosine=np.zeros(npix)
                self.sine=np.zeros(npix)
                counts,cosine,sine= self.counts,self.cosine,self.sine
                includes=r"""
                #include <stdio.h>
                #include <omp.h>
                #include <math.h>
                """
                code = """
                int i,pixel;
                for ( i=0;i<N;++i){
                    pixel=pixs(i);
                    if (pixel == -1) continue;
                    counts(pixel)   +=  w(i);
                    cosine(pixel)   +=  w(i)*cos(i);
                    sine(pixel)     +=  w(i)*sin(i);
                    cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                    sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                    sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                }
                """
                inline(code,['pixs','w','cos','sin','counts','cosine','sine','cos2','sin2','sincos','N'],
                extra_compile_args=['-march=native  -O3  -fopenmp ' ],verbose=1,
                support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)


    def repixelization(self):
        """
        Performs pixel reordering by excluding all the unbserved or
        pathological pixels.
        """
        n_new_pix=0
        n_removed_pix=0
        self.old2new=np.zeros(self.oldnpix,dtype=int)
        if self.pol==1:
            for jpix in xrange(self.oldnpix):
                if jpix in self.mask:
                    self.old2new[jpix]=n_new_pix
                    self.counts[n_new_pix]=self.counts[jpix]
                    self.obspix[n_new_pix]=self.obspix[jpix]
                    n_new_pix+=1
                else:
                    self.old2new[jpix]=-1
                    n_removed_pix+=1
            #resize array
            self.counts=np.delete(self.counts,xrange(n_new_pix,self.oldnpix))
        else:
            for jpix in xrange(self.oldnpix):
                if jpix in self.mask:
                    self.old2new[jpix]=n_new_pix
                    self.obspix[n_new_pix]=self.obspix[jpix]
                    self.cos2[n_new_pix]=self.cos2[jpix]
                    self.sin2[n_new_pix]=self.sin2[jpix]
                    self.sincos[n_new_pix]=self.sincos[jpix]
                    if self.pol==3:
                        self.counts[n_new_pix]=self.counts[jpix]
                        self.sine[n_new_pix]=self.sine[jpix]
                        self.cosine[n_new_pix]=self.cosine[jpix]
                    n_new_pix+=1
                else:
                    self.old2new[jpix]=-1
                    n_removed_pix+=1
            #resize
            self.cos2=np.delete(self.cos2,xrange(n_new_pix,self.oldnpix))
            self.sin2=np.delete(self.sin2,xrange(n_new_pix,self.oldnpix))
            self.sincos=np.delete(self.sincos,xrange(n_new_pix,self.oldnpix))
            if self.pol==3:
                self.counts=np.delete(self.counts,xrange(n_new_pix,self.oldnpix))
                self.sine=np.delete(self.sine,xrange(n_new_pix,self.oldnpix))
                self.cosine=np.delete(self.cosine,xrange(n_new_pix,self.oldnpix))
        c=bash_colors()
        print c.header("___"*30)
        print c.blue("Found %d pathological pixels\nRepixelizing  w/ %d pixels."%(n_removed_pix,n_new_pix))
        print c.header("___"*30)
        #resizing all the arrays
        self.obspix=np.delete(self.obspix,xrange(n_new_pix,self.oldnpix))
        self.__new_npix=n_new_pix

    def flagging_samples(self):
        """
        Flags the time samples related to bad pixels to -1.
        """
        N=self.nsamples
        o2n=self.old2new

        pixs=self.pixs
        code = """
	      int i,pixel;
          for ( i=0;i<N;++i){
            pixel=pixs(i);
            if (pixel == -1) continue;
            pixs(i)=o2n(pixel);
            }
        """
        inline(code,['pixs','o2n','N'],verbose=1,
		      extra_compile_args=['  -O3  -fopenmp ' ],
		      support_code = r"""
	               #include <stdio.h>
                   #include <omp.h>
	               #include <math.h>""",
            libraries=['gomp'],type_converters=weave.converters.blitz)
    def initializeweights(self,phi,w):
        """
        Pre-compute the quantitities needed for the implementation of :math:`(A^T A)`
        and to masks bad pixels.

        **Parameters**

        - ``counts`` :
            how many times a given pixel is observed in the timestream;
        - ``mask``:
            mask  either unobserved  (``counts=0``)  or   bad constrained pixels
            (see the ``pol=3,2`` following cases) ;
        - *If* ``pol=2``:
            the matrix :math:`(A^T A)`  is  symmetric and block-diagonal, each block
            can be written as :

            .. csv-table::

                ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            the determinant, the trace are therefore needed to compute the  eigenvalues
            of each block via the formula:

            .. math::
                \lambda_{min,max}= Tr(M)/2 \pm \sqrt{Tr^2(M)/4 - det(M)}

            being :math:`M` a ``2x2`` matrix.
            The eigenvalues are needed to define the mask of bad constrained pixels whose
            condition number is :math:`\gg 1`.

        - *If*  ``pol=3``*:
            each block of the matrix :math:`(A^T A)`  is a ``3 x 3`` matrix:

            .. csv-table::

                ":math:`n_{hits}`", ":math:`\sum_t cos 2 \phi_t`", ":math:`\sum_t sin 2 \phi_t`"
                ":math:`\sum_t cos 2 \phi_t`", ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                ":math:`\sum_t sin 2 \phi_t`",  ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            We then define the mask of bad constrained pixels by both  considering
            the condition number similarly as in the ``pol=2`` case and the pixels
            whose count is :math:`\geq 3`.

        """
        N=self.nsamples
        pixs=self.pixs
        if self.pol==1:
            self.counts=np.zeros(self.oldnpix)
            counts=self.counts
            includes=r"""
                    #include <stdio.h>
                    #include <omp.h>
                    #include <math.h>"""
            code = """
	              int i,pixel;
                  for ( i=0;i<N;++i){
                    pixel=pixs(i);
                    if (pixel == -1) continue;
                    counts(pixel)+=w(i);
                    }
                    """
            inline(code,['pixs','w','counts','N'],verbose=1,
                    extra_compile_args=['  -O3  -fopenmp ' ],
		            support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)
            self.mask=np.where(self.counts >0)[0]
        else:
            self.cos=np.cos(2.*phi)
            self.sin=np.sin(2.*phi)
            self.cos2=np.zeros(self.oldnpix)
            self.sin2=np.zeros(self.oldnpix)
            self.sincos=np.zeros(self.oldnpix)
            cos,sin         =   self.cos,self.sin
            cos2,sin2,sincos=   self.cos2,self.sin2,self.sincos
            if self.pol==2:
                includes=r"""
                        #include <stdio.h>
                        #include <omp.h>
                        #include <math.h>"""
                code = """
                      int i,pixel;
                      for ( i=0;i<N;++i){
                        pixel=pixs(i);
                        if (pixel == -1) continue;
                        cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                        sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                        sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                        }
                        """
                inline(code,['pixs','w','cos','sin','cos2','sin2','sincos','N'],verbose=1,
                        extra_compile_args=['  -O3  -fopenmp ' ],
                        support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)
            elif self.pol==3:
                self.counts=np.zeros(self.oldnpix)
                self.cosine=np.zeros(self.oldnpix)
                self.sine=np.zeros(self.oldnpix)
                counts,cosine,sine= self.counts,self.cosine,self.sine
                includes=r"""
                        #include <stdio.h>
                        #include <omp.h>
                        #include <math.h>"""
                code = """
                      int i,pixel;
                      for ( i=0;i<N;++i){
                        pixel=pixs(i);
                        if (pixel == -1) continue;
                        counts(pixel)+=w(i);
                        cosine(pixel)     +=  w(i)*cos(i);
                        sine(pixel)       +=  w(i)*sin(i);
                        cos2(pixel)     +=  w(i)*cos(i)*cos(i);
                        sin2(pixel)     +=  w(i)*sin(i)*sin(i);
                        sincos(pixel)   +=  w(i)*sin(i)*cos(i);
                        }
                        """
                inline(code,['pixs','w','cos','sin','counts','cosine','sine','cos2','sin2','sincos','N'],
                        extra_compile_args=['  -O3  -fopenmp ' ],verbose=1,
                        support_code = includes,libraries=['gomp'],type_converters=weave.converters.blitz)

            det=(self.cos2*self.sin2)-(self.sincos*self.sincos)
            tr=self.cos2+self.sin2
            sqrt=np.sqrt(tr*tr/4. -det)
            lambda_max=tr/2. + sqrt
            lambda_min=tr/2. - sqrt
            cond_num=np.abs(lambda_max/lambda_min)
            mask=np.where(cond_num<=self.threshold)[0]
            if self.pol==2:
                    self.mask=mask
            elif self.pol==3:
                mask2=np.where(self.counts>2)[0]
                self.mask=np.intersect1d(mask2,mask)
        print len(self.mask)
