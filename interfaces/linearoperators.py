import numpy as np
import linop.linop as lp
import linop.blkop as blk
import random as rd
import scipy.sparse.linalg as spla
from utilities_functions import *

class SparseLO(lp.LinearOperator):
    """
    Derived class from the one from the  ``LinearOperator`` in ``linop``.
    It constitutes an interface for dealing with the projection operator.

    Since this can be represented as a sparse matrix, it is initialized \
    by passing an array of pairs which resembles the  ``(i,j)`` positions \
    of the non-null elements of  the matrix.


    **Parameters**

    - ``n`` : {int}
        number of columns;
    - ``m`` : {int}
        number of rows;
    - ``pairs`` : {list of tuples, 2-d array }
         ``(i,j)`` positions of the non-null elements of  the matrix, ``A_(i,j)``;
    - ``pol`` : {int}
        process an intensity only map (``[default] pol=1``), intensity/polarization
        map (``pol=3``);
    - ``phi``: {array}
        array with polarization angles (if ``pol=3``).

    """

    def mult(self,v):
        """
        Performs the product of a sparse matrix ``A*v``,\
         with ``v`` a  ``numpy`` dense vector.

        It extracts the components of ``v`` corresponding  to the non-null \
        elements of the operator.
        """
        x=np.zeros(self.nrows)
        for (i,j) in self.pairs:
            x[i]+=v[j]

        return x

    def rmult(self,v):
        """
        Performs the product for the transpose operator ``A^T``.
        """
        x=np.zeros(self.ncols)
        for (i,j) in self.pairs:
            x[j]+=v[i]

        return x


    def mult_iqu(self,v):
        """
        Performs the product of a sparse matrix ``A*v``,\
        with ``v`` a  ``numpy`` dense vector containing the three Stokes [IQU].
        Compared to the operation ``mult`` this routine returns a TOD like vector,
        defined as:

        ``d_t= I_p + Q_p*cos(2*phi_t)+ U_p*sin(2*phi_t).``

        with ``p`` is the pixel observed at time ``t`` with polarization angle
        ``phi_t``.

        """
        x=np.zeros(self.nrows)
        for (i,j) in self.pairs:
            x[i]+=v[j]+v[j+1]*np.cos(2*self.phi[i])+v[j+2]*np.sin(2*self.phi[i])

        return x

    def rmult_iqu(self,v):
        """
        Performs the product for the transpose operator ``A^T`` to get a IQU-map
        like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        Values of the same pixel are stored in the memory contiguously.

        """
        x=np.zeros(self.ncols*3)
        for (i,j) in self.pairs:
            x[j]+=v[i]
            x[1+j]+=v[i]*np.cos(2*self.phi[i])
            x[2+j]+=v[i]*np.sin(2*self.phi[i])

        return x


    def __init__(self,n,m,pairs,phi=None,pol=1):
        self.ncols=n
        self.pol=pol
        self.nrows=m
        self.pairs=pairs

        if pol==3:
            self.__polarization='IQU'
            self.phi=phi
            super(SparseLO, self).__init__(nargin=pol*n,nargout=m, matvec=self.mult_iqu,
                                                symmetric=False, rmatvec=self.rmult_iqu )
        elif pol==1:
            self.__polarization='I'
            super(SparseLO, self).__init__(nargin=n,nargout=m, matvec=self.mult,
                                                symmetric=False, rmatvec=self.rmult )
        else:
            raise RuntimeError("No valid polarization key set!\t=>\tpol=%d \nPossible values are pol=%d(I),%d(IQU)."%(pol,1,3))
        print self.ncols,self.nrows
    def show(self):
        from scipy.sparse import coo_matrix
        i=[c[0] for c in self.pairs]
        j=[c[1] for c in self.pairs]
        A=coo_matrix((np.ones(self.nrows), (i,j)), shape=(self.nrows,self.pol*self.ncols)).toarray()
        import matplotlib.pyplot as plt
        plt.imshow(A)
        plt.show()

    @property
    def maptype(self):

        return self.__polarization


class ToeplitzLO(lp.LinearOperator):
    """
    Derived Class from a LinearOperator. It exploit the symmetries of an ``dim x dim``
    Toeplitz matrix.
    This particular kind of matrices satisfy the following relation:

    ``A_(i,j)=A_(i+1,j+1)=a_(i-j)``

    given the symmetry we further have:

    ``A_(i,j)=A_(i+1,j+1)=a_|i-j|``

    Therefore, it is enough to initialize ``A`` by mean of an array ``a`` of ``size = dim``.

    **Parameters**

    - ``a`` : {array, list}
        the array which resembles all the elements of the Toeplitz matrix;
    - ``size`` : {int}
        size of the block.

    """


    def mult(self,v):
        """
        Performs the product of a Toeplitz matrix with a vector ``x``.

        """
        val=self.array[0]
        y=val*v
        for i in xrange(1,len(self.array)):
            val=self.array[i]
            temp=val*v
            y[:-i]+=temp[i:]
            y[i:]+=temp[:-i]

        return y

    def __init__(self,a,size):

        super(ToeplitzLO, self).__init__(nargin=size,nargout=size, matvec=self.mult,
                                                symmetric=True )
        self.array=a

class BlockLO(blk.BlockDiagonalLinearOperator):
    """
    Derived class from the one in ``blkop`` module.
    It basically relies on the definition of a block diagonal operator,
    composed by ``nblocks`` diagonal operators.
    Each of them is a multiple  of the `Identity`` characterized by the ``nblocks`` values listed in ``t``.

    **Parameters**

    - ``blocksize`` : {int}
        The size of each diagonal block, it is : ``blocksize= n/nblocks``.
    - ``t`` : {array}
        noise values for each block in the operator
    - ``offdiag`` : {bool}
        it is strictly  related on the way you pass the array ``t``.

        - True : ``t`` is a list of array,\

                ``shape(t)= [nblocks,bandsize]``.
                In general ``bandsize!= blocksize`` in order
                to have a Toeplitz band diagonal operator.

        - False : ``t`` is a list with values that will define the diagonal operator.\

                    ``shape(t)=[nblocks]``.
                    Here for our convenience we consider
                    the diagonal of each block having the same ``double`` number.

    """
    def build_blocks(self):
        """
        Build each block of the operator either with or
        without off diagonal terms.

        """

        self.blocklist=[]
        if self.isoffdiag:
            self.blocklist = [ToeplitzLO(i,self.blocksize) for i in self.covnoise]

        if not self.isoffdiag:
            d=np.ones(self.blocksize)
            for i in self.covnoise:
                d.fill(i)
                self.blocklist.append(lp.DiagonalOperator(d))
                d=np.empty(self.blocksize)


    def __init__(self,blocksize,t,offdiag=None):
        self.__isoffdiag = offdiag
        self.blocksize=blocksize
        self.covnoise=t
        self.build_blocks()

        super(BlockLO, self).__init__(self.blocklist)

    @property
    def isoffdiag(self):
        return self.__isoffdiag


class InverseLO(lp.LinearOperator):
    """
    Construct the inverse operator of ``A``, ``A^-1`` as a linear operator.

    **Parameters**

    - ``A`` : {linear operator}
        the linear operator of the linear system to invert;
    - ``method`` : {function }
        the method to compute ``A^-1`` (see below);
    - ``P`` : {linear operator } (optional)
        the preconditioner for the computation of the inverse operator.

    """
    def mult(self,x):
        """
        It returns  ``y=A^-1*x`` by solving the linear system ``Ay=x``

        with a certain ``scipy`` routine  defined above as ``method``.

        """
        y,info = self.method(self.A,x,M=self.preconditioner)
        self.isconverged(info)
        return y

    def isconverged(self,info):
        """
        **Parameters**

        - ``info`` : {int}
            the output of the solver method.

        **Returns**

        - ``v`` : {bool}

        """
        self.__converged=info
        if info ==0:
            return True
        else :
            return False


    def __init__(self,A,method=None,preconditioner=None):
        super(InverseLO, self).__init__(nargin=A.shape[0],nargout=A.shape[1], matvec=self.mult,
                                                symmetric=True )
        self.A=A
        self.__method=method
        self.__preconditioner=preconditioner
        self.__converged=None

    @property
    def method(self):
        """
        The method to compute the inverse of A. \
        It can be any ``scipy`` solver, namely ``cg, gmres``.

        """
        return self.__method

    @property
    def converged(self):
        """
        **Returns**

        {int}

        It Provides convergence information:

        - 0 : successful exit;
        - >0 : convergence to tolerance not achieved, number of iterations;
        - <0 : illegal input or breakdown.

        """
        return self.__converged

    @property
    def preconditioner(self):
        """
        Preconditioner for the solver, for having fast computation.
        """
        return self.__preconditioner


from scipy.linalg import solve,lu

class CoarseLO(lp.LinearOperator):
    """
    This class contains all the operation involving the coarse operator ``E``.
    In this implementation ``E`` is always applied to a vector wiht
    its inverse : ``E^{-1}``.
    When initialized it performs an LU decomposition to accelerate the performances
    of the inversion.

    **Parameters**

    - ``Z`` : {list of arrays}
            deflation matrix;
    - ``AZ`` : {list of arrays}
            contains vectors ``A*Z_i``;
    - ``r`` :  {int}
            ``rank(Z)``, dimension of the deflation subspace.
    """

    def mult(self,v):
        """
        Perform the multiplication of the inverse coarse operator ``x=E^{-1} v``.
        It exploits the LU decomposition of ``E`` to solve the system ``Ex=v``.
        """
        y=solve(self.L,v,lower=True,overwrite_b=False )
        x=solve(self.U,y,overwrite_b=True)
        return x

    def __init__(self,Z,AZ,r):
        super(CoarseLO,self).__init__(nargin=r,nargout=r,matvec=self.mult,
                                            symmetric=True)
        M=[]
        #dot=get_blas_funcs('dot', (Z[0], AZ[0]))
        for j in AZ:
            M.append([scalprod(i,j) for i in Z])
        M=np.matrix(M)
        self.L,self.U=lu(M,permute_l=True,overwrite_a=True,check_finite=False)


class DeflationLO(lp.LinearOperator):
    """
    This class builds the Deflation operator (and its transpose)
    from the columns of the matrix ``Z``.

    **Parameters**

    - ``z`` : {list of arrays}
            columns of the deflation matrix.

    """

    def mult(self,x):
        """
        Performs the matrix vector multiplication   ``Z*x``
        with  ``dim(x) = rank(Z)``.

        """
        y=np.zeros(self.nrows)
        for i in xrange(self.ncols):
            y+=self.z[i]*x[i]
        return y

    def rmult(self,x):
        """
        Performs the product onto a ``N_pix`` vector with ``Z^T``.

        """
        return np.array( [scalprod(i,x) for i in self.z] )

    def __init__(self, z):
        self.ncols=len(z)
        self.nrows=len(z[0])
        self.z=z
        super(DeflationLO,self).__init__(nargin=self.ncols, nargout=self.nrows,
                                                matvec=self.mult, symmetric=False,
                                                rmatvec=self.rmult)
