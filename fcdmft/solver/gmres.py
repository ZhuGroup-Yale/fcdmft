import numpy as np
import scipy.sparse.linalg as spla

'''
GMRES/GCROT(m,k) for solving Green's function linear equations
'''

class GMRES(object):
    def __init__(self, A, b, x0, diag=None, tol=1e-3):
        n = max(b.shape)
        self.A = spla.LinearOperator((n,n), matvec=A)
        self.b = b.reshape(-1)
        self.x0 = x0.reshape(-1)
        self.tol = tol
        self.niter = 0
        if diag is None:
            self.M = None
        else:
            diag = diag.reshape(-1)
            Mx = lambda x: x/diag
            self.M = spla.LinearOperator((n,n), matvec=Mx)

    def solve(self):
        callback = None
        self.niter = 0
        def callback(rk):
            #print ("residual =", rk)
            self.niter += 1
        #self.x, self.info = spla.gmres(self.A, self.b, x0=self.x0, M=self.M,
        #                               maxiter=200, callback=callback, restart=40, tol=self.tol)
        self.x, self.info = spla.gcrotmk(self.A, self.b, x0=self.x0, M=self.M,
                                         maxiter=200, callback=callback, m=40, tol=self.tol)
        self.x = self.x.reshape(-1)
        if self.info > 0:
            print ("convergence to tolerance not achieved in", self.info, "iterations")
        return self.x


def setA(size, plus):
    A = np.zeros(shape=(size,size),dtype=complex)
    diag = np.zeros(shape=(size),dtype=complex)
    fac = 1.0
    for i in xrange(size):
      A[i,i] = 1.0 * fac + 6.0 * 1j * fac + plus + 30.*np.random.random()
      diag[i] = A[i,i] 
      if i+2 < size:
        A[i,i+2] = 1.0 * fac
      if i+3 < size:
        A[i,i+3] = 0.7 * fac
      if i+1 < size:
        A[i+1,i] = 3.0*1j * fac
    return A, diag


def main():
    size = 300
    A, diag = setA(size, 0.0+1j*0.0)
    b = np.random.rand(size) + 0j*np.random.rand(size)
    b /= np.linalg.norm(b)

    x0 = np.dot(np.linalg.inv(A),b)
    x0 += 1./1.*(np.random.rand(size) + 0j*np.random.rand(size))
    condition_number = np.linalg.cond(A)
    res = np.linalg.norm(b-np.dot(A,x0))
    finalx = np.dot(np.linalg.inv(A), b)
    print (" ::: Making A,b matrix :::")
    print ("  - condition number = %12.8f" % condition_number)
    print ("  - x0 residual      = %12.8f" % np.real(res))

    def multiplyA(vector, args=None):
        return np.dot(A,vector)

    gmin = GMRES(multiplyA, b, x0, diag)
    sol = gmin.solve()

    print ("|Ax-b| = ", np.linalg.norm(np.dot(A,sol) - b))

if __name__ == '__main__':
    main()
