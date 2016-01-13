from scipy.linalg import get_blas_funcs
import numpy as np
from utilities import *

def arnoldi(A, b, x0=None, tol=1e-5, maxiter=1000, inner_m=30 ):
    """
    Computes an orthonormal basis to get the approximated eigenvalues
    (Ritz eigenvalues) and eigenvector.

    The basis comes from a Gram-Schmidt orthonormalization of the Krylov
    subspace  defined as:

    .. math::

        K_m = span( b, Ab, ..., A^{m-1} b )

    at the :math:`m`-th iteration.

    **Parameters**

    - ``A`` : {sparse matrix , linear operator}
            matrix we want to approximate eigenvectors;
    - ``b`` : {array}
            array to build the Krylov subspace ;
    - ``x0`` : {array}
            initial guess vector to compute residuals;
    - ``tol`` : {float}
            tolerance threshold to the Ritz eigenvalue computation;
    - ``maxiter`` : {int}
            to validate the result one can compute ``maxiter`` times the
            eigenvalues, to seek the stability of the algorithm;
    - ``inner_m`` :  {int}
            maximum number of iterations within the Arnoldi algorithm,

            .. Warning::

                ``inner_m <=N_pix``

    **Returns**

    - ``w`` : {list of arrays}
            the orthonormal basis ``m x N_pix``;
    - ``h`` : {list of arrays}
            the elements of the :math:`H_m` Hessenberg matrix.
            At the ``m``-th iteration  :math:`h_m` has got :math:`m+1` elements.

    """
    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A.matvec
    axpy, dot, scal = None, None, None

    b_norm = norm2(b)
    if b_norm == 0:
        b_norm = 1

    for k_outer in xrange(maxiter):
        r_outer = b - matvec(x0)
        # -- determine input type routines
        if axpy is None:
            if np.iscomplexobj(r_outer) and not np.iscomplexobj(x0):
                x0 = x0.astype(r_outer.dtype)
            axpy, dot, scal = get_blas_funcs(['axpy', 'dot', 'scal'],
                                              (x0, r_outer))

        # -- check stopping condition
        r_norm = norm2(r_outer)
        if r_norm < tol * b_norm or r_norm < tol:
            print "r_norm < tol * b_norm or r_norm < tol"
            break

        # -- ARNOLDI ALGRITHM
        vs0 = scal(1.0/r_norm, r_outer)
        hs = []
        vs = [vs0]
        v_new = None

        for j in xrange(1, 1 + inner_m) :
            v_new=matvec(vs[j-1])
            v_new2 = v_new.copy()
            #     ++ orthogonalize
            hcur = []
            for v in vs:
                alpha = dot(v, v_new)
                hcur.append(alpha)
                v_new= axpy(v, v_new2, v.shape[0], -alpha)  # v_new -= alpha*v
            hcur.append(norm2(v_new))
            #       ++ normalize
            v_new = scal(1.0/hcur[-1], v_new)
            if hcur[-1] <= tol:
                print "--------------------------------------"
                print "Computed  %d Ritz eigenvalues within the tolerance %.1g "%(j,tol)
                print "--------------------------------------"


                hs.append(hcur)
                return vs,hs

            vs.append(v_new)
            hs.append(hcur)

            if j==inner_m:
                raise RuntimeError("Convergence not achieved within the Arnoldi algorithm")
                return None,None


def build_hess(h,m):
    """
    Compute  and store (as a Hessenberg matrix) the :math:`H_m` matrix from the
    output list ``h`` of the :func:`arnoldi` routine.

    **Parameters**

    - ``h`` : {list of arrays}
            matrix coefficients ;
    - ``m`` : {int}
            size of ``H``

    **Returns**

    - ``H`` :{numpy.matrix}

    """
    hess=np.zeros((m,m))
    for q in xrange(m-1):
        hess[:(q+2),q]=h[q]
    hess[:m,m-1]=h[-1][:m]

    return hess


def build_Z(z,y,w,eps):
    """
    Build the deflation matrix :math:`Z`. Its columns are the :math:`r`
    selected eigenvectors :math:`Z_i=w_m*y_i` s.t. their eigenvalues  :math:`z_i`
    are smaller than a certain threshold ``eps``.

    **Parameters**

    - ``z`` : {array}
        eigenvalues of :math:`H_m`;
    - ``y`` : {list of arrays}
        eigenvectors of :math:`H_m`;
    - ``w`` : {list of arrays}
        orthonormal basis (computed with the Arnoldi algorithm);
    - ``eps`` : {float}
        threshold to select the smallest eigenvalues.


    **Returns**

    - ``Z`` : {matrix}
            deflation subspace matrix;
    - ``r`` :  {int}
            :math:`rank(Z)`.

    """

    m=len(z)

    npix=len(w[0])

    select_eigvec=[]
    for i in xrange(m):
        if abs(z[i].real)<=eps:
            select_eigvec.append( y[i] )
    r=len(select_eigvec)
    if r==0 :
        raise RuntimeError("No Ritz eigenvalue are found smaller than fixed threshold %.1g "%eps)
    print "++++++++++++++++++++++++++++++++++++"
    print "Found  eigenvectors below the threshold %.1g!\nThe deflation subspace  has dim(Z)=%d "%(eps,r)
    print "++++++++++++++++++++++++++++++++++++"
    Z=dgemm(w,select_eigvec)
    return Z,r
