"""
    MUCM emulator for single outputs.

    Author: Sam Coveney
    Data: 21-04-2020

    TODO:
    * replace the weird MUCM nugget parameterization with a standard one? Why is MUCM inconsistent on this point?
      -> I have done this replacement... I assume it does not affect the ability to integrate out sigma successfully... Jeremy's thesis suggests it's okay

    * from "Uncertainty Analysis and other Inference Tools for Complex Computer Codes":
      "The first is that a(.) can usually only be `computed' subject to observation error. If we can assume that observation errors are normally distributed, then only a simple modification of the theory is needed. The main complication is that the error variance becomes another hyperparameter to be estimated."
    * from "Some Bayesian Numerical Analysis", in the section on smooth, O'Hagan expicitly says that "we simply add Vf (diagonal matrix of noise) to A as defined in 7)"

    NOTE: This means that in this 'integrate out sigma' formulation, we have sigma^2 * (A + nugget * I). In other words, the noise would be sigma^2 * nugget.
          This is an important caveat!


"""

from abc import ABC, abstractmethod

import numpy as np

import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import linalg
from scipy.optimize import minimize


#{{{ use '@timeit' to decorate a function for timing
import time
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        num = 30
        for r in range(num): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec for %d runs' % (f.__name__, te-ts, num) )
        return result
    return timed
#}}}


class AbstractModel(ABC):
    """Abstract class for MUCM emulator, requiring basis and kernel to be implemented elsewhere."""

    def __init__(self):
        """Initialize class."""

        super().__init__()


    # {{{ data handling including scalings
    def set_data(self, x, y):
        """Set data for interpolation, scaled and centred."""

        # save inputs
        self.x = x

        # save outputs, centred and scaled
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y = self.scale(y)

    def scale(self, y, stdev = False):
        """Scales and centres y data."""
        offset = 0 if stdev == True else self.y_mean
        return (y - offset) / self.y_std


    def unscale(self, y, stdev = False):
        """Unscales and uncenteres y data into original scale."""
        offset = 0 if stdev == True else self.y_mean
        return (y * self.y_std) + offset
    #}}}


    #{{{ abstract methods for H matrix and A matrix

    @abstractmethod
    def A_matrix(self):
        """Kernel matrix (not multiplied by signal variance)."""
        pass

    @abstractmethod
    def basis(self, X):
        """Defines the regressor basis matrix H."""
        pass

    #}}}


    #{{{ regressor basis matrix H
    def H_matrix(self, X = None):

        # this sets the H matrix for training inputs and saves it
        if X is None:
            save_self_H = True
            X = self.x
        else:
            save_self_H = False
        
        # make sure matrix X is 2D
        if X.shape[1] == 1: X = X[:,None]

        # build H matrix
        H = self.basis(X)
        
        if save_self_H:  # save H matrix between training points
            self.H = H
        else:  # return the H matrix for non-training points
            return H
    #}}}


    #{{{ loglikelihood
    def LLH(self, guess, get_sigma_beta = False):
        """
            MUCM - variance (sigma^2) and basis coefficients (beta) are integrated out
        """

        # set the hyperparameters
        guess = self.HP_untransform(guess)

        if self.nugget_train == True: # train the nugget
            self.lengthscale = guess[0:-1]
            nugget = guess[-1]
        else: # do not train the nugget
            self.lengthscale = guess[:]
            nugget = self.nugget


        A = self.A_matrix(nugget)
        y = self.y
        H = self.H
        n, q = self.x.shape[0], self.H.shape[1]

        ## calculate LLH
        #try:
        if True:
            L = linalg.cho_factor(A)        
            invA_y = linalg.cho_solve(L, y)
            invA_H = linalg.cho_solve(L, H)
            Q = (H.T).dot(invA_H)
            K = linalg.cho_factor(Q)
            B = linalg.cho_solve(K, (H.T).dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
            logdetA = 2.0*np.sum(np.log(np.diag(L[0])))
            logdetQ = 2.0*np.sum(np.log(np.diag(K[0])))

            invA_H_dot_B = invA_H.dot(B) # A^-1 H (H A^-1 H)^-1 H A^-1 y

            s2 = (1.0/(n-q-2.0))*(y.T).dot(invA_y-invA_H_dot_B)

            if get_sigma_beta: return [s2, B]

            LLH = -0.5*(-(n-q)*np.log(s2) - logdetA - logdetQ)

        else:
    
            # second test with cholesky directly
            L = np.linalg.cholesky(A)        

            L_y = scipy.linalg.solve_triangular(L, y, lower = True)

            # Q = H A-1 H
            L_H = scipy.linalg.solve_triangular(L, H, lower = True)
            Q = L_H.T.dot(L_H)
            LQ = np.linalg.cholesky(Q) # NOTE: should really be renamed to something other than K

            logdetA = 2.0*np.sum(np.log(np.diag(L)))
            logdetQ = 2.0*np.sum(np.log(np.diag(LQ)))
            
            # calculate B = y.T A^-1 H (H A^-1 H)^-1 H A^-1 y
            # note that 'beta' is (H A^-1 H)^-1 H A^-1 y
            tmp = L_H.T.dot(L_y) # H A^-1 y
            tmp = scipy.linalg.solve_triangular(LQ, tmp, lower = True)
            B = np.dot(tmp.T, tmp)

            s2 = (1.0/(n-q-2.0))*( np.dot(L_y.T, L_y) - B )

            # NOTE: beta is actually (H A^-1 H)^-1 H A^-1 y, so with this method I can't just extract beta, will need a separate routine
            if get_sigma_beta:
                c = scipy.linalg.solve_triangular(LQ, tmp, lower = True)
                beta = scipy.linalg.solve_triangular(LQ.T, c, lower = False)
                    
                return [s2, beta]


            LLH = -0.5*(-(n-q)*np.log(s2) - logdetA - logdetQ)

            
        return LLH

#        except np.linalg.linalg.LinAlgError as e:
#            print("  WARNING: Matrix not PSD for", guess, ", not fit.")
#            return None
#
#        except ValueError as e:
#            print("  WARNING: Ill-conditioned matrix for", guess, ", not fit.")
#            return None
    #}}}
    

    #{{{ optimization
    def optimize(self, nugget, restarts = 10):
        """Optimize the hyperparameters.
        
           Arguments:
           nugget -- value of the nugget, if None then train nugget, if number then fix nugget
           restart -- how many times to restart the optimizer (default 10).

        """

        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str("{:1.3f}".format(x))[:5] % x

        # initial guesses for hyperparameters
        dguess = np.random.uniform(self.HP_transform(0.1), self.HP_transform(10), size = restarts).reshape([1,-1]).T
        if self.x.shape[1] > 1:
            for i in range(1,self.x.shape[1]):
                dg = np.random.uniform(self.HP_transform(0.1), self.HP_transform(10), size = restarts).reshape([1,-1]).T
                dguess = np.hstack([dguess, dg])
        nguess = np.random.uniform(self.HP_transform(1e-4), self.HP_transform(1e-2), size = restarts).reshape([1,-1]).T

        hdr = "Restart | "
        for d in range(dguess.shape[1]):
            hdr = hdr + " len " + " | "
        guess = np.hstack([dguess])

        # nugget
        self.nugget = nugget

        if nugget is None:  # nugget not supplied; we must train on nugget
            self.nugget_train = True
            guess = np.hstack([guess, nguess])
            self.nugget_index = guess.shape[1] - 1
            hdr = hdr + " nug " + " | " 
        else:
            nugget = np.abs(nugget)
            self.nugget_train = False

        # run optimization code
        print("Optimizing Hyperparameters...\n" + hdr)

        # create the H matrix
        self.H_matrix()

        for ng, g in enumerate(guess):
            optFail = False
            #try:
            if True:
                bestStr = "   "
                #res = minimize(self.LLH, g, args = (False), method = 'Nelder-Mead') 
                res = minimize(self.LLH, g, args = (False), method = 'L-BFGS-B') 

                if np.isfinite(res.fun):
                    try:
                        if res.fun < bestRes.fun:
                            bestRes = res
                            bestStr = " * "
                    except:
                        bestRes = res
                        bestStr = " * "
                else:
                    bestStr = " ! "

                print(" {:02d}/{:02d} ".format(ng + 1, restarts),
                      "| %s" % ' | '.join(map(str, [fmt(i) for i in self.HP_untransform(res.x)])),
                      "| {:s} llh: {:.3f}".format(bestStr, -1.0*np.around(res.fun, decimals=4)))

            #except TypeError as e:
            #    optFail = True

        self.s2, self.beta = self.LLH(bestRes.x, get_sigma_beta = True)
        self.HP = self.HP_untransform(bestRes.x)
        self.lenthscale = self.HP[0:self.x.shape[1]]
        self.nugget = self.HP[-1] if nugget is None else nugget 

        print("\nHyperParams:", ", ".join(map(str, [fmt(i) for i in self.HP])), \
              "\n  (s2: {:f})".format(self.s2),
              "\n  (beta:", ", ".join(map(str, [fmt(i) for i in self.beta])), ")", 
              "\n  (nugget: {:f})".format(self.nugget), 
              "\n  (noise var = s2*nugget: {:f})".format(self.nugget*self.s2) )

    #}}}


    #{{{ posterior prediction
    def posterior(self, X, full_covar = False):
        """Posterior prediction at points X."""

        if X.shape[1] != self.x.shape[1]:
            print("[ERROR]: inputs features do not match training data.")
            return

        y = self.y
        A = self.A_matrix(self.nugget)
        self.H_matrix()
        H = self.H
        s2 = self.s2
        beta = self.beta

        A_cross = self.A_matrix_cross(X, self.x, self.nugget)

        # NOTE: calculate pointwise posterior only
        #A_pred = self.A_matrix_cross(X, X, self.nugget)
        A_pred = 1.0

        H_pred = self.H_matrix(X)

        try:
            L = linalg.cho_factor(A)        
            invA_H = linalg.cho_solve(L, H)
            Q = H.T.dot(invA_H)
            K = linalg.cho_factor(Q)
            T = linalg.cho_solve(L, y - H.dot(beta))
            R = H_pred - A_cross.dot( invA_H )

            mean = H_pred.dot( beta ) + ( A_cross ).dot(T)

            # one way of doing it
            #var = s2 * ( A_pred - (A_cross).dot( linalg.cho_solve(L, A_cross.T) ) + R.dot( linalg.cho_solve(K, R.T) ) )

            # NOTE: calculate pointwise posterior only
            tmp_1 = np.einsum("ij, ji -> i", A_cross, linalg.cho_solve(L, A_cross.T) )
            tmp_2 = np.einsum("ij, ji -> i", R, ( linalg.cho_solve(K, R.T) ) )
            tmp = tmp_1 + tmp_2
            var = s2 * ( A_pred - tmp )

            return self.unscale(mean, stdev = False), self.unscale(np.sqrt(var), stdev = True) 

        except np.linalg.linalg.LinAlgError as e:
            print("ERROR:", e)
            return None

    #}}}


# types of emulator (different combinations of kernels and basis functions)

#{{{ various subclasses

#{{{ RBF abstract class
class RBF(AbstractModel):
    """ A matrix: RBF kernel, used for other classes that implement a specific basis.
        Note that the nugget is 'within' the covariance, so will be multiplied by the variance.
    """

    def HP_transform(self, HP):
        return np.log(HP)


    def HP_untransform(self, HP):
        return np.exp(HP)


    def A_matrix(self, nugget):
        """A matrix between training data inputs."""

        w = 1.0 / self.lengthscale
        A = pdist(self.x*w,'sqeuclidean')
        A = np.exp(-A)
        A = squareform(A)

        np.fill_diagonal(A , 1 + nugget)

        return A


    def A_matrix_cross(self, xi, xj, nugget):
        """A matrix more generally."""

        w = 1.0 / self.lengthscale
        A = cdist(xi*w, xj*w,'sqeuclidean')
        A = np.exp(-A)

        return A
 
    @abstractmethod
    def basis(self, X):
        pass

#}}}

#{{{ RBF + linear basis
class RBF_linear(RBF):
    """ A matrix: RBF kernel. H matrix: linear.
    """

    def basis(self, X):
        """Linear mean function."""
        H = np.hstack( [np.ones((X.shape[0],1)) , X] )
        return H

#}}}

#{{{ RBF + custom H matrix
class RBF_custom(RBF):
    """ A matrix: RBF kernel. H matrix: custom, user must implement basis by attaching a function to it, or defining it here.
    """

    def basis(self):
        pass

#}}}

#}}}

