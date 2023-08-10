# the following is from github.com/jcouto/btss
from statsmodels.stats.proportion import proportion_confint
from scipy.special import erfc # import the complementary error function
from scipy.special import erf # import the error function
# This has example ways to fit the psychometric function and getting confidence intervals
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize
import numpy as np

# weibull fit
def weibull(bias, slope, gamma1, gamma2, X):
    ''' weibull function with lapse rates
    '''
    return gamma1 + (1. - gamma1 - gamma2) * (erf((X - bias) / slope) + 1.) / 2. +1e-9

def cumulative_gaussian(alpha,beta,gamma,lmbda, X):
    '''
    Evaluate the cumulative gaussian psychometric function.
       alpha is the bias (left or right)
       beta is the stepness
       gamma is the left handside offset
       lmbda is the right handside offset
      
    Adapted from the Palamedes toolbox 
    Joao Couto - Jan 2022    
    '''
        
    from scipy.special import erfc # import the complementary error function
    return  gamma + (1 - gamma - lmbda)*0.5*erfc(-beta*(X-alpha)/np.sqrt(2))+1e-9    

# log likelihood when using minimize
def neg_log_likelihood_error(func, parameters, X, Y):
    '''
    Compute the log likelihood

    'func' is the (psychometric) function 
    'parameters' are the input parameters to 'func'
    'Y' is the binary response (correct = 1; incorrect=0)
    Joao Couto - Jan 2022    
    '''

    pX = func(*parameters, X)*0.99 + 0.005  # the predicted performance for X from the PMF
    # epsilon to prevent error in log(0)
    val = np.nansum(Y*np.log(pX) + (1-Y)*np.log(1-pX))
    return -1*val

def compute_proportions(stim_values, response_values):
    '''
    Computes the proportion of responses to each stimulus value.
Returns: 
    - stims    - unique stimulus intensities
    - p_side   - proportion of trials to the side
    - ci_side  - confidance intervals from binomial distribution with the wilson method
    - n_obs    - number of observations (trials for each)
    - n_side   - number of trials to the specific side


Joao Couto - Jan 2023
    '''
    
    stims = np.unique(stim_values)
    p_side = np.zeros_like(stims,dtype=float) 
    ci_side = np.zeros((len(stims),2),dtype=float)
    n_obs = np.zeros_like(stims,dtype=float)
    n_side = np.zeros_like(stims,dtype=float)
    for i,intensity in enumerate(stims):
        # number of times the subject licked to one of the sides 
        cnt = np.sum(response_values[stim_values == intensity]) 
        nobs = np.sum(stim_values == intensity) # number of observations (ntrials)
        n_obs[i] = nobs
        n_side[i] = cnt
        p_side[i] = cnt/nobs
        ci_side[i] = proportion_confint(cnt,nobs,method='wilson') # 95% confidence interval
    return stims,p_side,ci_side,n_obs,n_side

# statsmodels gives confidence and p values for the fit
class PsychometricRegression(GenericLikelihoodModel):
    '''
        Fits a psychometric with constraints function (constrained weibull e.g.)
        
        This is part of the tools for github.com/jcouto/btss
        
        Inputs:
            endog - are the response values for each trial
            exog  - are the stim intensities for each trial
            func  - the function to fit; default weibull with lapses
            bound - the constraints to the fit; default [(min(x),max(x)),(0.01,1000),(0,1),(0,1)]
            startpar_function - a function to get the start guess for the fit
            parnames - the names of the fit parameters
            
        Usage:
        
        ft = PsychometricRegression(response_values.astype(float),
                                    exog = stim_values.astype(float))
        res = ft.fit(min_required_stim_values = min_required_stim_values, full_output=True)
        print(res.summary())
            
        Joao Couto - Feb 2023
        '''
    def __init__(self, endog, exog, func = None, bounds = None,startpar_function = None,parnames = None, **kwds):
        '''
        Fits a psychometric with constraints function (constrained weibull e.g.)
        
        This is part of the tools for github.com/jcouto/btss
        
        Inputs:
            endog - are the response values for each trial
            exog  - are the stim intensities for each trial
            func  - the function to fit; default weibull with lapses
            bound - the constraints to the fit; default [(min(x),max(x)),(0.01,1000),(0,1),(0,1)]
            startpar_function - a function to get the start guess for the fit
            parnames - the names of the fit parameters
            
        Usage:
        
        ft = PsychometricRegression(response_values.astype(float),
                                    exog = stim_values.astype(float))
        res = ft.fit(min_required_stim_values = min_required_stim_values, full_output=True)
        print(res.summary())
            
        Joao Couto - Feb 2023
        '''

        super(PsychometricRegression, self).__init__(endog, exog, **kwds)
        if not func is None:
            self.fit_function = func
            self.bounds = None
            self.get_start_params = startpar_function
            if not parnames is None:
                self.exog_names[:] = parnames
        else:
            self.fit_function = cumulative_gaussian
            self.exog_names[:] = ['bias','sensitivity','gamma1','gamma2']
            self.bounds = [(np.min(self.exog[:,0]),np.max(self.exog[:,0])),
                           (0.001,0.3),
                           (0,0.5),(0,0.5)]
            self.get_start_params  = lambda x,y: [0,1./np.max(x),y[0],1-y[-1]]
            
    
    def loglikeobs(self,params):
        pX = self.fit_function(*params, self.exog[:,0])
        pX[pX<=0] = 0.0001
        pX[pX>=1] = 0.9999
        ii = np.where(np.isfinite(pX))
        # the predicted performance for X from the PMF
        val = np.nansum(self.endog[ii]*np.log(pX[ii]) + (1-self.endog[ii])*np.log(1-pX[ii]))
        return val
    
    def fit(self, start_params=None,
            maxiter=100000,
            maxfun=10000,
            method = 'lbfgs', # Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm because it can be constrained
            min_required_stim_values=6, **kwds):

        self.stims, self.p_side, self.ci_side, self.n_obs,self.n_side = compute_proportions(self.exog[:,0], self.endog)
        if len(self.stims) < min_required_stim_values:
            return None
        
        if start_params == None:
            # Reasonable starting values
            if not self.get_start_params is None:
                start_params = self.get_start_params(self.stims, self.p_side)
        if start_params is None:
            raise(ValueError("Need to provide start parameters or a function to compute them."))
        self.df_null = 0
        self.k_constant = len(start_params)
        self.df_resid = len(self.endog)-len(start_params)
        return super(PsychometricRegression, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            method = method,
            bounds = self.bounds,
            disp = False,
            **kwds)
        
# main function to fit and compute statistics
def fit_psychometric(stim_values, response_values,
                     func = cumulative_gaussian,    # to fit the psychometric
                     min_required_stim_values = 6,  # min values required to fit the function
                     method = 'PsychometricRegression'):#'Nelder-Mead'):#'L-BFGS-B'):
    '''
    Fits a psychometric curve and computes points
    
    Joao Couto - Jan 2023
    '''
    if method == 'PsychometricRegression':
        # use PsychometricRegression (default weibull)
        ft = PsychometricRegression(response_values.astype(float),
                                    exog = stim_values.astype(float))
        res = ft.fit(min_required_stim_values = min_required_stim_values, full_output=True)
        if res is None:
            fit_res = None
            params = None
        else:
            fit_res = res
            params = res.params
        return dict(stims = ft.stims,
                    p_side = ft.p_side,
                    p_side_ci = ft.ci_side,
                    n_side = ft.n_side,
                    n_obs = ft.n_obs,
                    fit_params = params,
                    fit = fit_res,
                    function = ft.fit_function)
    
    stims,p_side,ci_side,n_obs,n_side = compute_proportions(stim_values, response_values)
    fit_res = None
    params = None
    if len(stims) >= min_required_stim_values and not func is None:

        opt_func = lambda pars: neg_log_likelihood_error(func, pars, 
                                                     stim_values,
                                                     response_values)
        # x0 is the initial guess for the fit, it is an important parameter
        x0 = [0.,0.1,p_side[0],1 - p_side[-1]]
        bounds = [(stims[0],stims[-1]),(0.0001,10),(0,0.7),(0,.7)]
        #import warnings
        #warnings.filterwarnings('ignore')
        options = dict(maxiter = 500*len(x0))
        if 'Neder' in method:
            options['adaptive'] = True
        fit_res = minimize(opt_func, x0,
                           options = options,
                           bounds = bounds, method = method)
        params = fit_res.x
    return dict(stims = stims,
                p_side = p_side,
                p_side_ci = ci_side,
                n_side = n_side,
                n_obs = n_obs,
                fit_params = params,
                fit = fit_res,
                function = func)
