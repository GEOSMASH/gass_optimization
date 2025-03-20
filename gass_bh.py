import numpy as np
import pandas as pd
import geopandas as gpd

import statsmodels.api as sm
from spglm.iwls import _compute_betas
from mgwr.search import golden_section
import scipy
from spglm import family
from scipy.optimize import basinhopping

from smoother import ConstantTerm, LinearTerm, KernelSmoothing, DistanceSmoothing
from copy import deepcopy

import pkg_resources

class GASS:
    def __init__(self, y, *args, family = 'Gaussian', constant = True):
        self.y = y
        self.family = family.lower()
        self.args = args # Input model terms
        self.term_mapping = {} # Dictionary to store the mapping of each covariate in 'X' to its corresponding term
        self.constant = constant # Intercept
        self.num_constant_term = 0 # NO. of intercept, it should be one.
        self.num_linear_terms = 0 # NO. of linear terms
        self.initial_sigmas = [] # initalized hyperparameters, `sigma` is used to refer to any hyperparamter in different spatial smoothing types
        self.sigmas = [] 
        self.z = None
        self.initial_X = None 
        self.final_X = None
        self.AWCI_sigmas = None
        self.RBCI_sigmas = None
        self.CI_betas = None
        self.fitted_y = None
        self.residuals = None
        self.std_err = None
        self.tvals = None
        self.zvals = None # Poisson
        self.pvals = None
        self.AIC = None
        self.log_likelihood = None
        self.Deviance = None # Poisson
        self.R_squared = None # Gaussian
        self.R_squared_CS = None # Poisson
        self.R_squared_McFadden = None # Poisson
        self.percent_deviance = None # Poisson

        self.mgwr_version = pkg_resources.parse_version(pkg_resources.get_distribution("mgwr").version)  # Store mgwr version once
        
        self._initialize()

    def _initialize(self):
        initial_X_matrices = []
        current_col_index = 0 # Current column index in initial_X
        
        if self.constant == True:
            constant = ConstantTerm(self.y.shape[0])
            initial_X_matrices.append(constant.X)
            current_col_index = 1
            self.num_constant_term = 1
            self.term_mapping [0] = (type(constant).__name__, constant)  
        
        for arg_idx, arg in enumerate(self.args):
            num_columns = 0  # Number of columns this arg will add to initial_X

            if isinstance(arg, LinearTerm):
                initial_X_matrices.append(arg.X)
                num_columns = arg.X.shape[1]  
                self.num_linear_terms += num_columns 
                
            elif isinstance(arg, DistanceSmoothing):
                initial_X_matrices.append(arg.cal(arg.initial_value))
                num_columns = arg.cal(arg.initial_value).shape[1]
                self.initial_sigmas.append(arg.initial_value)
                
            elif isinstance(arg, KernelSmoothing):
                initial_X_matrices.append(arg.cal(arg.initial_k))
                num_columns = arg.cal(arg.initial_k).shape[1]
                self.initial_sigmas.append(arg.initial_k)
                
            # elif isinstance(arg, SATerm):
            #     initial_X_matrices.append(arg.cal(arg.initial_sigma))
            #     num_columns = arg.cal(arg.initial_sigma).shape[1]
            #     self.initial_sigmas.append(arg.initial_sigma)
                
            else:
                raise ValueError(f"Unsupported term type: {type(arg)}")

            # Record the term mapping for the new columns
            for col in range(current_col_index, current_col_index + num_columns):
                self.term_mapping [col] = (type(arg).__name__, arg)   # storing index and type name

            current_col_index += num_columns  # update current column index

        # Concatenate terms
        self.initial_X = np.hstack(initial_X_matrices)
        
    def fit(self, input_y = None, max_iter = 50, crit_threshold = 1e-8, printed = False, verbose = False):
        
        # Ensure valid family
        supported_families = ['gaussian', 'poisson']
        if self.family not in supported_families:
            raise ValueError(f"Invalid `family` type: {self.family}. Supported options: {supported_families}")
        else:
            print(f"Running fit for {self.family.capitalize()} ...")

        # Initialize y and X
        self._X = self.initial_X.copy() # store updated smoothed X values per iteration
        y = self.y.copy()
        
        # Designed for residual bootstrap confidence interval calculation
        if input_y is not None:
            y = input_y.copy()

        # Initialize (hyper)parameters
        sigmas = self.initial_sigmas.copy()
        betas = np.zeros(self._X.shape[1]) * 1.000001

        # Initialize family operations: adjust y
        family_ops = self._family_handler(y)
        v, mu = family_ops['init']()

        # Initialize iteration settings
        crit = 1e6
        n_iter = 0

        # Outer loop of local scoring algorithm
        while crit > crit_threshold and n_iter < max_iter:

            # Adjust y and calculate statistical weights for weighted least square (WLS)
            w, z = family_ops['adjust'](v, mu)
            
            # Apply square root weights for WLS(wz ~ wx) in backfitting
            w = np.sqrt(w)
            wx = np.multiply(self._X, w.reshape(-1,1))
            wz = np.multiply(z.reshape(-1,1), w.reshape(-1,1))
            self.w = w.copy()

            # Inner loop: backfitting
            n_betas, tmp_sigmas = self._backfit(wz, wx, sigmas, max_iter = max_iter, tol = crit_threshold, printed = printed, verbose = verbose) 
            sigmas = deepcopy(tmp_sigmas)

            # Update statistical weights
            v, mu = family_ops['update'](self._X, n_betas)
            
            # Compute convergence criterion 
            num = np.sum((n_betas - betas)**2) / len(y)
            den = np.sum(np.sum(n_betas, axis=1)**2)
            crit = (num / den)**0.5
            betas = n_betas
            
            n_iter += 1 # increment the iteration counter

        # Store final results
        self.coefficients = betas
        self.sigmas = sigmas
        self.final_X = self._X
        self.z = z
        self.w = w
        self.wz = wz
        self.wx = wx
    
        pass
        
    def _backfit(self, y, X, sigs, max_iter = 50, tol = 1e-8, printed = False, verbose = False):
        _,k = X.shape
        
        # Compute initial parameter estimatesas
        betas = _compute_betas(y, X)
        
        # Compute initial additive terms, y hat and redisuals
        XB = np.multiply(betas.T, X)
        yhat = np.dot(X, betas)
        err = y.reshape((-1, 1)) - yhat 

        # Initialize iterations setting
        scores = []
        delta = 1e6
        tmp_sigs = sigs

        for n_iter in range(1, max_iter + 1):
            # Initialize the iteration
            new_XB = np.zeros_like(X)
            params = np.zeros_like(betas)

            for j in range(k):
                
                temp_y = XB[:, j].reshape((-1, 1)) + err.reshape((-1, 1))
                temp_X = X[:, j].reshape((-1, 1))
                type_name, term_instance = self.term_mapping[j]  
            
                # Identify spatial smooothers and implement optimization
                if type_name not in ['LinearTerm', 'ConstantTerm']:
                    
                    # Objective function of optimization
                    def aic_func(x):
                        x_scalar = x.item() if isinstance(x, np.ndarray) else x[0] if isinstance(x, list) else x  
                        newX = np.hstack((X[:, np.arange(X.shape[1]) != j], term_instance.cal(x_scalar) * self.w))
                        aic = sm.GLM(y, newX, family = sm.families.Gaussian()).fit().aic
                        return aic 
                    
                    # Prepare arguments for basin hopping
                    
                    x0 = (term_instance.upper_bound-term_instance.lower_bound)/2
                    x0 = int(x0) if term_instance.int_score else x0
                    search_range = [(term_instance.lower_bound, term_instance.upper_bound)]
                    minimizer_kwargs = {
                        "method": 'L-BFGS-B',
                        "bounds": search_range
                    }
                    
                    bh_result = basinhopping(aic_func, x0, minimizer_kwargs=minimizer_kwargs)
                    sig, aic = bh_result.x[0], bh_result.fun
                    sig = int(sig) if term_instance.int_score else sig
                    
                    tmp_sigs[j-self.num_linear_terms-self.num_constant_term] = sig # store the updated sig
                    sv = term_instance.cal(sig) # the new smoothed values of Xj with the updated sig
                    self._X[:, j] = sv.flatten() # store the new smoothed values of Xj
                    X[:, j] = (sv * self.w).flatten() # update Xj value
                    temp_X = (sv * self.w).flatten().reshape((-1,1)) # update temp_X
                    
                    if printed:
                        print(sig, aic)

                # Compute parameter for the variable Xj with the updated sig
                beta = _compute_betas(temp_y, temp_X)
                
                # Update y hat, additive terms and residuals
                yhat = np.dot(temp_X, beta)
                new_XB[:, j] = yhat.flatten()
                err = (temp_y - yhat).reshape((-1, 1))
                params[j, :] = beta[0][0]

            # Compute convergence criterion  
            num = np.sum((XB-new_XB)**2)
            den = 1 + np.sum(np.sum(XB, axis=1)**2)
            score = (num / den)
            scores.append(deepcopy(score))
            delta = score
            
            # Update additive terms with the final values from this iteration
            XB = new_XB

            if verbose:
                print("Current iteration:", n_iter, ",SOC:", np.round(score, 8))
            if delta < tol:
                break
        
        return params, tmp_sigs

    def _family_handler(self, y):
        """Handles all family-specific operations."""
        
        supported_families = {
            'gaussian': family.Gaussian(),
            'poisson': family.Poisson()
        }
    
        fam = supported_families.get(self.family)
        if fam is None:
            raise ValueError(f"Unsupported family: {self.family}")
    
        # Gaussian
        if isinstance(fam, family.Gaussian):
            def init_gaussian():
                return None, None
        
            def adjust_response_gaussian(v, mu):
                w = np.ones(len(y)).reshape(-1, 1)  
                z = y.reshape(-1, 1)  
                return w, z
        
            def update_statistical_weights_gaussian(X, betas):
                return None, None

            def infer_gaussian(X, betas):
                n, k = X.shape
                yhat = np.dot(X, betas).flatten()
                residuals = y - yhat
                s2 = np.sum(residuals ** 2) / (X.shape[0] - X.shape[1])  # Variance
                
                var_beta = s2 * np.linalg.inv(X.T @ X).diagonal()
                std_err = np.sqrt(var_beta)
                r2 = 1 - (np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))
                logLm = -n/2 * (1 + np.log(2*np.pi)) - n/2 * np.log(s2)
                aic =  2*k - 2*logLm
                
                return {
                    'fitted_y': yhat,
                    'residuals': residuals,
                    'std_err': std_err,
                    'R_squared': r2,
                    'log_likelihood': logLm,
                    'AIC': aic
                }
        
            return {'init': init_gaussian, 'adjust': adjust_response_gaussian, 'update': update_statistical_weights_gaussian, 'infer': infer_gaussian}
            
        # Poisson
        elif isinstance(fam, family.Poisson):
            offset = np.ones(len(y))
            
            def init_poisson():
                y_off = fam.starting_mu(y / offset)
                v = fam.predict(y_off).reshape(-1, 1)
                mu = fam.starting_mu(y).reshape(-1, 1)
                return v, mu
    
            def adjust_response_poisson(v, mu):
                w = fam.weights(mu)
                z = v + (fam.link.deriv(mu) * (y.reshape(-1, 1) - mu))
                return w, z
    
            def update_statistical_weights_poisson(X, betas):
                v = np.dot(X, betas)  
                mu = fam.fitted(v) * offset.reshape(-1, 1)
                return v, mu

            def infer_poisson(X, betas):
                n, k = X.shape
                eta = np.dot(X, betas)
                fitted_y = np.exp(eta).flatten()
    
                residuals = (y - fitted_y) / np.sqrt(fitted_y)
                V_inv = fitted_y.flatten()
                cov_beta = np.linalg.inv((X.T * V_inv) @ X)
                std_err = np.sqrt(np.diag(cov_beta))
    
                lambda_null = np.mean(y)
                logL0 = np.sum(-lambda_null + y * np.log(lambda_null) - scipy.special.gammaln(y + 1))
                logLm = np.sum(-fitted_y + y * np.log(fitted_y) - scipy.special.gammaln(y + 1))
                aic = 2 * k - 2 * logLm

                ratio = y / fitted_y
                ratio = np.where(ratio == 0, 1, ratio)
                deviance = 2 * np.sum(y * np.log(ratio) - (y - fitted_y))

                ratio2 = y / lambda_null
                ratio2 = np.where(ratio2 == 0, 1, ratio2)
                null_deviance = 2 * (np.sum(y * np.log(ratio2) - (y - lambda_null)))
                model_deviance = 2 * np.sum(y - fitted_y + y * np.log(ratio))
                
                prct_deviance = 1 - (model_deviance / null_deviance)
                r2_cs = 1 - np.exp(2 * (logL0 - logLm) / len(y))
                r2_mf = 1 - logLm / logL0

                zval = betas.flatten() / std_err
                
                return {
                    'fitted_y': fitted_y,
                    'residuals': residuals,
                    'std_err': std_err,
                    'log_likelihood': logLm,
                    'AIC': aic,
                    'Deviance': deviance,
                    'percent_deviance': prct_deviance,
                    'R_squared_CS': r2_cs,
                    'R_squared_McFadden': r2_mf,
                    'zval': zval
                }
    
            return {'init': init_poisson, 'adjust': adjust_response_poisson, 'update': update_statistical_weights_poisson, 'infer': infer_poisson}

    def _calculate_CI_betas(self, betas, se_beta, n, k, dist="t"):
        if dist == "t":
            critical_value = scipy.stats.t.ppf(1 - 0.05 / 2, df = n - k)
        elif dist == "z":
            critical_value = scipy.stats.norm.ppf(1 - 0.05 / 2)
        else:
            raise ValueError("Invalid distribution type for confidence interval calculation.")
        
        coefs_lower = betas.flatten() - critical_value * se_beta
        coefs_upper = betas.flatten() + critical_value * se_beta
        return list(zip(coefs_lower, coefs_upper))
    
    def inference(self):
        # Ensure the family type
        if self.family is None:
            raise ValueError("The `family` parameter must be specified (e.g., 'Gaussian' or 'Poisson').")

        family_ops = self._family_handler(self.y)
        infer_results = family_ops['infer'](self.final_X, self.coefficients)
        n, k = self.final_X.shape
        
        # Store results
        self.fitted_y = infer_results['fitted_y']
        self.residuals = infer_results['residuals']
        self.std_err = infer_results['std_err']
        self.log_likelihood = infer_results['log_likelihood']
        self.AIC = infer_results['AIC']

        # Compute confidence intervals of betas
        self.CI_betas = self._calculate_CI_betas(self.coefficients, self.std_err, n, k, dist="t" if self.family == "gaussian" else "z")
        
        # Compute p-values
        self.tvals = self.coefficients.flatten() / self.std_err
        self.pvals = 2 * (1 - scipy.stats.t.cdf(np.abs(self.tvals), df = n - k)) if self.family == 'gaussian' else 2 * (1 - scipy.stats.norm.cdf(np.abs(self.tvals)))

        # Store results for differnet families
        if self.family == 'gaussian':
            self.R_squared = infer_results['R_squared']
        
        elif self.family == 'poisson':
            self.zvals = infer_results['zval']
            self.Deviance = infer_results['Deviance']
            self.percent_deviance = infer_results['percent_deviance']
            self.R_squared_CS = infer_results['R_squared_CS']
            self.R_squared_McFadden = infer_results['R_squared_McFadden']
    
    def calculate_AWCI_sigmas(self, level = 0.95):
        
        self.AWCI_sigmas = []
        for tidx, tsig in enumerate(self.sigmas):
            
            tsig_idx = int(tidx + self.num_linear_terms + self.num_constant_term)
            tsig_term_instance = self.term_mapping[tsig_idx][1]
            
            # create an array of candidate sigmas
            tsig_b4 = np.arange(tsig_term_instance.lower_bound, tsig, tsig_term_instance.CI_step)
            tsig_af = np.arange(tsig, tsig_term_instance.upper_bound, tsig_term_instance.CI_step)
            tsig_candidates = np.hstack((tsig_b4, tsig_af)).flatten()
            
            tsig_aics = []
            for sig in tsig_candidates:
                wx = np.hstack((self.wx[:, np.arange(self.wx.shape[1]) != tsig_idx], tsig_term_instance.cal(sig) * self.w))
                aic = sm.GLM(self.wz, wx, family = sm.families.Gaussian()).fit().aic
                tsig_aics.append((sig, aic))
                
            tsig_awdf = pd.DataFrame(tsig_aics, columns=['Sigma', 'AIC'])  

            minAIC = np.min(tsig_awdf.AIC)
            deltaAICs = tsig_awdf.AIC - minAIC
            awsum = np.sum(np.exp(-0.5 * deltaAICs))
            tsig_awdf = tsig_awdf.assign(AW = np.exp(-0.5 * deltaAICs)/awsum)
            tsig_awdf = tsig_awdf.sort_values(by = 'AW',ascending=False)
            tsig_awdf = tsig_awdf.assign(cumAW = tsig_awdf.AW.cumsum())

            index = len(tsig_awdf[tsig_awdf.cumAW < level]) + 1
            tsig_min = tsig_awdf.iloc[:index,:].Sigma.min()
            tsig_max = tsig_awdf.iloc[:index,:].Sigma.max()
            
            self.AWCI_sigmas.append((round(tsig_min, 4), round(tsig_max,4)))
            
        pass
    
    def calculate_RBCI_sigmas(self, level=0.95, max_iter = 100, crit_threshold = 1e-8, printed = False):
        
        if self.fit_function is None:
            raise ValueError("No fit function has been set. Please call fit_Gaussain or fit_Poisson before calling this method.")

        fitted_y = self.fitted_y.copy()
        residuals = self.y.reshape(-1,1) - self.fitted_y
        
        self.RBCI_sigmas = []
        sigdicts = {}
        lower = (1 - level) * 100 / 2.0
        upper = 100 - lower

        sigdicts = {i: [] for i in range(len(self.sigmas))}

        for i in range(max_iter):
            
            np.random.seed(i)
            bootstrap_residuals = np.random.choice(residuals[:, 0], size=len(residuals), replace=True).reshape(-1, 1)
            bootstrap_y = (fitted_y + bootstrap_residuals).flatten()
            tgass = deepcopy(self)

            tgass.fit(input_y = bootstrap_y, crit_threshold = crit_threshold)
            
            for tidx, tsig in enumerate(tgass.sigmas):
                sigdicts[tidx].append(tsig)
                
            if printed:
                print(i)

        for siglist in sigdicts.values():
            sigdf = pd.DataFrame(siglist)
            sigdf.columns = ['Sigma']
            sigdf = sigdf.sort_values(by=['Sigma'])

            minSig = np.percentile(sigdf, lower)
            maxSig = np.percentile(sigdf, upper)

            self.RBCI_sigmas.append((round(minSig, 4), round(maxSig, 4)))
