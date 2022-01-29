from sklearn.base import BaseEstimator
import numpy as np
from numpy.linalg import norm as vector_norm

from scipy.special import logsumexp
from scipy.linalg import pinvh

def to_spherical_coords(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[np.newaxis, :]
  
    # normalize vectors to unit length
    Xunit = (X.T / np.sqrt(np.diag(np.dot(X,X.T)))).T
  
    # compute spherical coordinates (assuming x up)
    Gamma = np.zeros((Xunit.shape[0], 2))
    Gamma[:,0] = np.arctan2(Xunit[:,2],Xunit[:,1]) # azimuth
    Gamma[:,1] = np.arccos(Xunit[:,0]) # inclination
    
    return Gamma
    
def to_cartesian_coords(Gamma):
    Gamma = np.asarray(Gamma)
    if Gamma.ndim == 1:
        Gamma = Gamma[np.newaxis, :]
        
    X = np.zeros( (Gamma.shape[0], 3) )
    # Gamma = [ [phi_0, theta_0], ... ]
    X[:,0] = np.cos(Gamma[:,1])
    X[:,1] = np.sin(Gamma[:,1])*np.cos(Gamma[:,0])
    X[:,2] = np.sin(Gamma[:,1])*np.sin(Gamma[:,0])
    
    return X


def average_spherical_coords(Gamma, weights=None):
    Gamma = np.asarray(Gamma)
    if Gamma.ndim == 1:
        Gamma = Gamma[np.newaxis, :]
    gamma = np.zeros( (1, 2) )
    gamma[0,0] = np.arctan2(np.average (np.sin(Gamma[:,0]), 0, weights), np.average (np.cos(Gamma[:,0]), 0, weights))
    gamma[0,1] = np.arctan2(np.average (np.sin(Gamma[:,1]), 0, weights), np.average (np.cos(Gamma[:,1]), 0, weights))
    
    return gamma
    
    
def log_kent_density(X, Gs, dispers, kappas, betas):
    """Compute the log probability under a Kent (Fisher-Bingham) distribution.
      
    Parameters
    ----------
    `X` : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    `G` : array_like, shape (n_samples, n_features)
        
    `dispers`: array_like
        List of n_components dispersion matrices for each Kent distribution.
    `kappa` : array, shape (`n_components`, )
        Concentration parameter for each mixture component.

    `beta` : array, shape (`n_components`, )
        Ovalness parameter for each mixture component.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Kent distribution.
    """
    lpr = np.zeros( (Gs.shape[0], X.shape[0], ) )
    
    for k, G, S, kappa, beta in zip(range(Gs.shape[0]), Gs,dispers,kappas,betas):
        # Are parameters valid?
        if kappa <= 0 or beta <0 or kappa < 2*beta:
            raise ValueError("WARNING: Invalid distribution parameters. " +
                    "kappa: %f, beta: %f" % (kappa, beta))
            
        Z = -kappa + np.log(np.sqrt(kappa**2 - 4*beta**2)) - np.log(2*np.pi)
        lpr[k,:] = Z + kappa*np.dot(X, G[:,0]) + beta*np.dot(X, G[:,1])**2 - beta*np.dot(X, G[:,2])**2
        
    # FIXME remove this comment?
    # normalize lpr 
    # Normalization is not very exact; we can artificially normalize
    # by selecting the highest probability and dividing by it
    # (i.e. subtracting in log-space)
    #lpr -= np.max(lpr) 
    
    return lpr.T

class KentMixture(BaseEstimator):
    """Kent Mixture Model (also known as Fisher-Bingham Mixture Model)

    Representation of a Kent mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a KMM distribution.

    The Kent distribution is the direction analogue to a bivariate Gaussian
    distribution. It allows to model a distribution of directions on
    a 3 dimensional sphere. Note that this algorithm is restricted to
    3 dimensional data as moment estimation for higher dimensional data
    is too difficult. 
    
    A higher-dimensional directional distribution would be the 
    von-Mises-Fisher-distribution (not implemented here). However, 
    this distribution is constrained to isotropic distributions, 
    i.e. it is the analogue of a multivariate isotropic Gaussian.
    The Kent distribution, on the other hand, allows for representing
    non-isotropic directional distributions.

    For estimation of moments of a single distribution see 
      Kent, J.T. (1982) The Fisher-Bingham distribution on the sphere., 
        J. Royal. Stat. Soc., 44:71-80.
    The EM algorithm for fitting a mixture of Kent distributions is described in
      Peel, D., Whiten, WJ., McLachlan, GJ. (2001) Fitting mixtures of 
        Kent distributions to aid in joint set identification. 
        J. Am. Stat. Ass., 96:56-63
    
    The most detailed description of the here implemented algorithm is found in
      McLachlan, G., & Peel, D. (2004). Finite mixture models. 
        John Wiley & Sons.

    Currently the cluster centers are initialized randomly.
    Initialization should be improved, e.g. by running spherical 
    k-means (i.e. k-means with dot-product distance measure)
    as done in the GMM node.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    thresh : float, optional
        Convergence threshold.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        number of initializations to perform. the best results is kept

    Attributes
    ----------
    `Gamma_` : array, shape (`n_components`, n, 3)
        Data in spherical coordinates
        
    `weights_` : array, shape (`n_components`,)
        Mixing weights for each mixture component.

    `G_` : array, shape (`n_components`, 3, 3)
        Orientation matrix for each mixture component.
        1st column: Mean direction (pole)
        2nd column: Major axis
        3rd column: Minor axis

    `S_` : array, shape (`n_components`, 3, 3)
        Sample dispersion matrix for each mixture component.
        The dispersion is the spherical analogue to the covariance
        of a Gaussian distribution.

    `kappa_` : array, shape (`n_components`, )
        Concentration parameter for each mixture component.
        Concentration is a reciprocal measure of dispersion, i.e. the
        closer kappa is to zero, the more uniform the distribution is.
        If kappa is large, the distribution becomes very concentrated
        around the mean.

    `beta_` : array, shape (`n_components`, )
        Ovalness parameter for each mixture component. 
        2*beta < kappa must hold to ensure correct behavior.

    'gamma_': array, shape (`n_components`, 2)
        Sample mean (of data in spherical coordinates)

    'R_': array, shape (`n_components`, )
        Norm of sample mean (of data in cartesian coordinates)

    `V_` : array, shape (`n_components`, 3, 3)
    `B_` : array, shape (`n_components`, 3, 3)
    `H_` : array, shape (`n_components`, 3, 3)
    `K_` : array, shape (`n_components`, 3, 3)

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.


    See Also
    --------
      examples/mixture/plot_kmm.py
    
    """
          
    def __init__(self, n_components = 1, random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1):
        self.n_components = n_components
        self.thresh = thresh
        self.min_covar = min_covar
        self.random_state = random_state

        self.n_iter = n_iter
        self.n_init = n_init

        self.weights_ = np.ones(self.n_components) / self.n_components

        # flag to indicate exit status of fit() method: converged (True) or
        # n_iter reached (False)
        self.converged_ = False
       
        self.eps = np.finfo(float).eps

        self.verbose_ = False
        #self.verbose_ = True
                
    
    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob: array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities: array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != 3:
            raise ValueError('the shape of X  is not compatible (not 3)')
            
        lpr = (log_kent_density(
                X, self.G_, self.S_, self.kappa_, self.beta_)
               + np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])

        return logprob, responsibilities
            
            
    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        logprob, responsibilities = self.eval(X)
        return responsibilities

    def sample(self, n_samples=1, random_state=None):
        raise NotImplementedError("KentMixture.sample is not implemented" )
        
    def fit(self, X):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string ''. Likewise, if you
        would like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """

        ## initialization step
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] < self.n_components:
            raise ValueError(
                'Kent Mixture Estimation estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        if X.shape[1] != 3:
            raise ValueError(
                'Kent Mixture Estimation only works for 3D cartesian vectors, only %dD' %
                (X.shape[1]))
        
        # normalize
        for i in range(X.shape[0]):
            X[i,:] = X[i,:] / np.sqrt(np.dot(X[i,:].T, X[i,:]))

        self.Gamma_ = to_spherical_coords(X)
        if self.verbose_:
          print ("Gamma:\n%s" % str(self.Gamma_))
                
        max_log_prob = - np.infty

        if self.n_init < 1:
            raise ValueError('KMM estimation requires at least one run')
        
        n = X.shape[0]
        
        for __n_init in range(self.n_init):
            # init intermediate data structures
            self.G_ = np.zeros( (self.n_components, 3, 3) )
            self.S_ = np.zeros( (self.n_components, 3, 3) )
            
            self.kappa_ = np.ones( (self.n_components, ) )
            self.beta_ = np.zeros( (self.n_components, ) )
            
            # choose k random vectors as initialization
            # TODO better initialization
            for k in range(self.n_components):
                self.G_[k,:,0] = np.random.randn(3)
                self.G_[k,:,0] /= np.linalg.norm(self.G_[k,:,0])
            
            if self.verbose_:
                np.set_printoptions(suppress=True)
                np.set_printoptions(precision=3)
                print ("kappa: " + str(self.kappa_))
                print ("beta: " + str(self.beta_))
                print ("G: " + str(self.G_))
                print ("S: " + str(self.S_))
            
            log_likelihood = []
            # reset self.converged_ to False
            self.converged_ = False
            
            for i in range(self.n_iter):
                if self.verbose_:
                    print ("----------------------")
                    print ("iteration: %d" % i)
                
                # Expectation step
                curr_log_likelihood, responsibilities = self.score_samples(X)
                log_likelihood.append(curr_log_likelihood.sum())

                # Check for convergence.
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
                        self.thresh:
                    self.converged_ = True
                    break
                
                if self.verbose_:
                    np.set_printoptions(precision=3)
                    np.set_printoptions(suppress=True)
                    print ("curr_log_likelihood: %s" % str(curr_log_likelihood))
                    print ("responsibilities: %s" % str(responsibilities.T))

                # Maximization step
                try:
                    self._do_mstep(X, self.Gamma_, responsibilities, self.min_covar)
                except ValueError as e:
                    print (e)
                    log_likelihood.append(-np.infty)
                    break
                    
                if self.verbose_:
                    np.set_printoptions(precision=3)
                    print ("new means: %s" % str([ self.G_[x,:,0] for x in range(self.G_.shape[0]) ]))

            if self.n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights_,
                                   'G': self.G_,
                                   'S': self.S_,
                                   'kappa': self.kappa_,
                                   'beta': self.beta_,
                                   }
                    if self.verbose_:
                        print ("n_iter %d has loglikelihood: %f" % (__n_init, max_log_prob) )
                else:
                    if self.verbose_:
                        print ("n_iter %d has loglikelihood: %f" % (__n_init, log_likelihood[-1]) )

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.weights_ = best_params['weights']
            self.G_ = best_params['G']
            self.S_ = best_params['S']
            self.kappa_ = best_params['kappa']
            self.beta_ = best_params['beta']
            
        return self
    
    def get_params(self):
        return {'G_': self.G_, 
                'S_': self.S_, 
                'kappa_': self.kappa_, 
                'beta_': self.beta_,
                'weights_': self.weights_
               }
    
    def set_params(self, param_dict):
        self.G_ = param_dict['G_']
        self.S_ = param_dict['S_']
        self.kappa_ = param_dict['kappa_']
        self.beta_ = param_dict['beta_']
        self.weights_ = param_dict['weights_']

    def _do_mstep(self, X, Gamma, responsibilities, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weights.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * self.eps)

        self.weights_ = (weights / (weights.sum() + 10 * self.eps) + self.eps)

        n = X.shape[0]

        # auxiliary matrices
        self.gamma_ = np.zeros( (self.n_components, 2) )
        self.R_ = np.zeros( (self.n_components, ) )
        
        self.V_ = np.zeros( (self.n_components, 3, 3) )
        self.B_ = np.zeros( (self.n_components, 3, 3) )
        self.H_ = np.zeros( (self.n_components, 3, 3) )
        self.K_ = np.zeros( (self.n_components, 3, 3) )
        
        for i in range(self.n_components):
            if np.all(responsibilities[:,i] == 0):
                # cluster is "dead", i.e. empty
                raise ValueError("Cluster %d is empty." % i)

            self.gamma_[i,:] = average_spherical_coords (Gamma, responsibilities[:,i])            
            if self.verbose_:
                print ("gamma: %s / %s" % (str(self.gamma_[i,:]), str(to_cartesian_coords(self.gamma_[i,:]))))
            
            self.R_[i] = np.linalg.norm( np.average(X, 0, responsibilities[:,i]) )

            self.S_[i] = np.zeros( (3,3) )
            for j in range(n):
                xxT = np.outer(X[j,:], X[j,:])
                self.S_[i] += xxT*responsibilities[j,i]
            self.S_[i] /= n

            phi, theta = self.gamma_[i,:]
            self.H_[i] = np.array([
                [np.cos(theta), - np.sin(theta), 0],
                [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
                [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],  
            ])

            self.B_[i] = np.dot(np.dot(self.H_[i].T, self.S_[i]), self.H_[i])
            alpha_hat = 0.5 * np.arctan2(2.*self.B_[i,1,2], self.B_[i,1,1] - self.B_[i,2,2])

            self.K_[i,:,:] = np.array([
                [1.,0.,0.],
                [0., np.cos(alpha_hat), -np.sin(alpha_hat)],
                [0., np.sin(alpha_hat), np.cos(alpha_hat)],
            ])

            self.G_[i,:,:] = np.dot(self.H_[i], self.K_[i])
            self.V_[i,:,:] = np.dot(np.dot(self.G_[i].T, self.S_[i]), self.G_[i])
            W = self.V_[i,1,1] - self.V_[i,2,2]

            # concentration
            self.kappa_[i] = 1./( 2.*(1.-self.R_[i]) - W) + 1./(2*(1.-self.R_[i]) + W)
            # ovalness
            self.beta_[i] = 0.5 * (1./(2.*(1.-self.R_[i]) - W) - 1./(2*(1.-self.R_[i]) + W))

            if self.verbose_:
                print ("k: %d" % i)
                print ("R: " + str(self.R_[i]))
                print ("W: " + str(W))

            if W < self.eps:
                # singularity! this means this cluster has collapsed
                # on one data point
                raise ValueError("Cluster %d has collapsed on one data point." % i)
          
        return weights



