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

import numpy as np

import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import linalg
from scipy.optimize import minimize

from functools import partial
import inspect

import jax.numpy as jnp
from jax import jit, jacfwd, grad, value_and_grad, config
from jax.scipy.linalg import solve_triangular as jst
config.update('jax_enable_x64', True)



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


class Emulator:
    """ Emulator class.

        Initialize example (RBF and Linear can be important from this package):

        model = Emulator(kernel = RBF(dim = 5), basis = Linear)

    """


    #{{{ init
    def __init__(self, kernel, basis):

        try:
            if inspect.isclass(kernel): raise ValueError
        except ValueError as e:
            print("[ERROR: ValueError]: 'kernel' must be a kernel class instance, e.g. RBF(dim = 3) not RBF. Exiting.")
            exit()

        self.kernel = kernel 
        self.basis = basis
    #}}}


    # {{{ data handling including scalings
    def set_data(self, x, y):
        """Set data for interpolation, scaled and centred."""

        # save inputs
        self.x = x

        # save outputs, centred and scaled
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y = self.scale(y)

        # create the H matrix for training data
        self.H_matrix()


    def scale(self, y, stdev = False):
        """Scales and centres y data."""
        offset = 0 if stdev == True else self.y_mean
        return (y - offset) / self.y_std


    def unscale(self, y, stdev = False):
        """Unscales and uncenteres y data into original scale."""
        offset = 0 if stdev == True else self.y_mean
        if len(y.shape) == 1:
            return (y * self.y_std) + offset

        if len(y.shape) > 1 and stdev == True: # this is for unscaling the variance
            return ( (y * self.y_std**2) )

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


    #{{{ transformations for optmizing nugget
    @partial(jit, static_argnums=(0))
    def nugget_transform(self, x): return jnp.log(x)

    @partial(jit, static_argnums=(0))
    def nugget_untransform(self, x): return jnp.exp(x)
    #}}}


    #{{{ loglikelihood
    @partial(jit, static_argnums=(0, 2))
    def LLH(self, guess, fixed_nugget): #, get_sigma_beta = False):
        """
            MUCM - variance (sigma^2) and basis coefficients (beta) are integrated out
        """

        # set the hyperparameters
        if guess.shape[0] > self.kernel.dim: # training on nugget
            HP = self.kernel.HP_untransform(guess[0:-1])
            nugget = self.nugget_untransform(guess[-1])
        else: # fixed nugget
            HP = self.kernel.HP_untransform(guess)
            nugget = fixed_nugget

        A = self.kernel.A_matrix(HP, self.x, self.x) + nugget*jnp.eye(self.x.shape[0])

        y = self.y
        H = self.H
        n, q = self.x.shape[0], self.H.shape[1]

        ## calculate LLH
        try:
        
            # second test with cholesky directly
            L = jnp.linalg.cholesky(A)        

            L_y = jst(L, y, lower = True)

            # Q = H A-1 H
            L_H = jst(L, H, lower = True)
            Q = jnp.dot(L_H.T, L_H)
            LQ = jnp.linalg.cholesky(Q)

            logdetA = 2.0*jnp.sum(jnp.log(jnp.diag(L)))
            logdetQ = 2.0*jnp.sum(jnp.log(jnp.diag(LQ)))
            
            # calculate B = y.T A^-1 H (H A^-1 H)^-1 H A^-1 y
            #                   beta = (H A^-1 H)^-1 H A^-1 y
            tmp = jnp.dot(L_H.T, L_y) # H A^-1 y
            tmp_2 = jst(LQ, tmp, lower = True)
            B = jnp.dot(tmp_2.T, tmp_2)

            s2 = (1.0/(n-q-2.0))*( jnp.dot(L_y.T, L_y) - B )

            llh = -0.5*(-(n-q)*jnp.log(s2) - logdetA - logdetQ)

            #print("guess:", guess)
            #print("LLH:", LLH)

            return llh

        except jnp.linalg.linalg.LinAlgError as e:
            print("  WARNING: Matrix not PSD for", guess, ", not fit.")
            return None

        except ValueError as e:
            print("  WARNING: Ill-conditioned matrix for", guess, ", not fit.")
            return None

    #}}}
    

    #{{{ optimization
    def optimize(self, nugget, restarts = 10):
        """Optimize the hyperparameters.
        
           Arguments:
           nugget -- value of the nugget, if None then train nugget, if number then fix nugget
           restart -- how many times to restart the optimizer (default 10).

        """

        guess = self.kernel.HP_guess(num = restarts)

        # nugget stuff
        if nugget is None:  # nugget not supplied; train nugget
            fixed_nugget = 0.0
            nguess = np.random.uniform(self.nugget_transform(1e-4), self.nugget_transform(1e-2), size = restarts).reshape([1,-1]).T
            guess = np.append(guess, nguess, axis = 1)
        else:  # nugget supplied; fix nugget
            fixed_nugget = np.abs(nugget)


        # construct the header for printing optimzation results
        # FIXME: it would be better to round up to the number of figures we're actually printing
        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str("{:1.3f}".format(x))[:5] % x
        hdr = "\033[1mRestart | "
        for d in range( self.kernel.dim ): hdr = hdr + " HP{:d} ".format(d) + " | "
        if nugget is None:  hdr = hdr + " nug " + " | "
        hdr = hdr + "\033[0m"
        print("Optimizing Hyperparameters...\n" + hdr)


        #{{{ jit autograd setup for LLH
        # NOTE: To use jax with lbfgs, need to wrap returns to cast to numpy array - https://github.com/google/jax/issues/1510
        #       e.g.
        #       grad_llh = grad(self.LLH)
        #       grad_func = lambda x: np.array(jit(grad_llh)(x))  # not sure if this extra jit here is needed
        #       Also, to use 32-bit, need to cast resulting answer to 64-bit; however, results are poor/wrong with 32-bit

        # value and grad
        new_llh = value_and_grad(self.LLH)
        #newer_llh = lambda x, y: [ np.array(val) for val in new_llh(x, y) ]
        newer_llh = lambda x, y: [ np.array(val) for val in jit(new_llh)(x, y) ]

        #}}}
        

        for ng, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "

                #res = minimize(self.LLH, g, method = 'L-BFGS-B', jac = grad_func) # for jax with separate func and grad 
                res = minimize(newer_llh, g, method = 'L-BFGS-B', jac = True, args = (fixed_nugget)) # for jax with joint func and grad

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
                      "| %s" % ' | '.join(map(str, [fmt(i) for i in self.kernel.HP_untransform(res.x)])),
                      "| {:s} llh: {:.3f}".format(bestStr, -1.0*np.around(res.fun, decimals=4)))

            except TypeError as e:
                optFail = True

        # save results of optimization
        if nugget is None:
            self.HP = self.kernel.HP_untransform(bestRes.x[:-1])
            self.nugget = self.nugget_untransform(bestRes.x[-1])
        else:
            self.HP = self.kernel.HP_untransform(bestRes.x)
            self.nugget = fixed_nugget

        self.store_values() # save other useful things

        print("\nHyperParams:", ", ".join(map(str, [fmt(i) for i in self.HP])), \
              "\n  (s2: {:f})".format(self.s2),
              "\n  (beta:", ", ".join(map(str, [fmt(i) for i in self.beta])), ")", 
              "\n  (nugget: {:f})".format(self.nugget), 
              "\n  (noise var = s2*nugget: {:f})".format(self.nugget*self.s2) )

    #}}}

    
    #{{{ store some important variables
    def store_values(self):
        """Calculate and save some important values."""

        self.A = self.kernel.A_matrix(self.HP, self.x, self.x) + self.nugget*jnp.eye(self.x.shape[0])

        y, H = self.y, self.H
        n, q = self.x.shape[0], self.H.shape[1]

        L = jnp.linalg.cholesky(self.A)        
        L_y = jst(L, y, lower = True)
        L_H = jst(L, H, lower = True)
        Q = jnp.dot(L_H.T, L_H) # Q = H A-1 H
        LQ = jnp.linalg.cholesky(Q)

        tmp = jnp.dot(L_H.T, L_y) # H A^-1 y
        tmp_2 = jst(LQ, tmp, lower = True)
        B = jnp.dot(tmp_2.T, tmp_2) # B = y.T A^-1 H (H A^-1 H)^-1 H A^-1 y

        s2 = (1.0/(n-q-2.0))*( jnp.dot(L_y.T, L_y) - B )
        beta = jst(LQ.T, tmp_2, lower = False) # beta = (H A^-1 H)^-1 H A^-1 y

        # save important values
        self.L, self.LQ = L, LQ
        self.s2, self.beta = s2, beta

        return
    #}}}


    #{{{ posterior prediction
    def posterior(self, X, full_covar = False):
        """Posterior prediction at points X."""

        if X.shape[1] != self.x.shape[1]:
            print("[ERROR]: inputs features do not match training data.")
            return

        A_cross = self.kernel.A_matrix(self.HP, X, self.x)

        H_pred = self.H_matrix(X)

        H, y, beta, s2 = self.H, self.y, self.beta, self.s2
        L, LQ = self.L, self.LQ

        try:

            T = linalg.cho_solve((L, True), y - H.dot(beta))

            invA_H = linalg.cho_solve((L, True), H)
            R = H_pred - A_cross.dot( invA_H )

            mean = H_pred.dot( beta ) + ( A_cross ).dot(T)

            if full_covar:
                A_pred = self.kernel.A_matrix(self.HP, X, X)

                tmp_1 = np.dot(A_cross, linalg.cho_solve((L, True), A_cross.T) )
                tmp_2 = np.dot(R, linalg.cho_solve((LQ, True), R.T) )
                tmp = tmp_1 + tmp_2
                var = s2 * ( A_pred - tmp )

                return self.unscale(mean, stdev = False), self.unscale(var, stdev = True) 

            else:
                A_pred = 1.0

                tmp_1 = np.einsum("ij, ji -> i", A_cross, linalg.cho_solve((L, True), A_cross.T) )
                tmp_2 = np.einsum("ij, ji -> i", R, linalg.cho_solve((LQ, True), R.T) )
                tmp = tmp_1 + tmp_2
                var = s2 * ( A_pred - tmp )

                return self.unscale(mean, stdev = False), self.unscale(np.sqrt(var), stdev = True) 


        except np.linalg.linalg.LinAlgError as e:
            print("ERROR:", e)
            return None

    #}}}


