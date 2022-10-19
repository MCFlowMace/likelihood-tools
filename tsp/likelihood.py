
"""

Author: F. Thomas
Date: September 22, 2022

"""

import numpy as np
from scipy.stats import norm
from scipy.stats import chi2

from abc import ABC, abstractmethod

class LikelihoodModel(ABC):
    
    def __init__(self, f):
        self.f = f
        
    @abstractmethod
    def log_likelihood_pdf(self):
        pass
        
    @abstractmethod
    def gen_noise(self, n):
        pass
    
class GaussianLikelihoodModel(LikelihoodModel):
    
    def __init__(self, f, sigma):
        LikelihoodModel.__init__(self, f)
        self.sigma = sigma
        
    #def likelihood_pdf(self):
    #    return lambda theta,x: norm.pdf(x,scale=self.sigma, loc=self.f(*theta))
    def log_likelihood_pdf(self):
        return lambda theta,x: -0.5*((x - self.f(*theta))/self.sigma)**2
        
    def gen_noise(self, n):
        return np.random.normal(loc=0, scale=self.sigma, size=n)

def get_best_fit_vals(scan_vals, ll_vals):
    
    max_ind = np.argmax(ll_vals)
    
    print(max_ind, scan_vals.shape[0])
    
    if max_ind==0 or max_ind==scan_vals.shape[0]-1:
        raise ValueError('Likelihood maximum not in scanning range')
    
    best_fit = scan_vals[max_ind]
    border = ll_vals[max_ind] - 0.5
    
    left_ind = np.searchsorted(ll_vals[:max_ind], border, side='right')
    left_val = scan_vals[:max_ind][left_ind]
    right_ind = np.searchsorted(ll_vals[max_ind:][::-1], border, side='right')
    right_val = scan_vals[max_ind:][::-1][right_ind]
    
    return best_fit, left_val, right_val
    

def get_best_fit(llh, axes):
    max_llh, max_ind = get_max_llh(llh)
    
    best_fit = np.empty(len(axes))
    for i, ax in enumerate(axes):
        best_fit[i] = ax[max_ind[i]]
        
    return best_fit, max_llh


class LikelihoodScan:
    
    def __init__(self, truth, llh, axes, ax_names, fixed_view_parameters):
        
        self.truth = truth
        self.llh = llh
        self.axes = axes
        self.ax_names = ax_names
        self.fixed_view_parameters = fixed_view_parameters
        
    def make_view(self, view=None):
        
        if view is not None:
            view_slices = self.make_view_slices(view)
            view_ax_ind = self.make_view_ax_ind(view)

            truth = [t for i, t in enumerate(self.truth) if view_ax_ind[i]]
            llh = self.llh[tuple(view_slices)]
            axes = tuple(ax for i, ax in enumerate(self.axes) if view_ax_ind[i])
            ax_names = [n for i, n in enumerate(self.ax_names) if view_ax_ind[i]]
            fixed_view_parameters = {name: (self.axes[i][view_slices[i]], i) for i, name in enumerate(self.ax_names) if not view_ax_ind[i]}
            
            if self.fixed_view_parameters is not None:
                fixed_view_parameters_old = self.fixed_view_parameters.copy()
                fixed_view_parameters_old.update(fixed_view_parameters)
                fixed_view_parameters = fixed_view_parameters_old

            return LikelihoodScan(truth, llh, axes, ax_names, fixed_view_parameters)
        else:
            return self
        
    #~ @classmethod
    #~ def make_scan(cls, data, n_grid, scan_window, ax_names, model, truth):
        
        #~ llh, axes = scan_likelihood(data, truth, scan_window, n_grid, model)
        
        #~ view = tuple(None if n>0 else 'default' for n in n_grid)
        
        #~ return cls(truth, llh, axes, ax_names, None).make_view(view)
        
    def make_view_slices(self, view):
        
        slices = []

        for i, ax in enumerate(self.axes):
            if view[i] is None:
                slices.append(slice(0, None, 1))
            elif view[i]=="default":
                index = int(len(ax)/2)
                slices.append(index)
            else:
                index = self.find_closest_ax_ind(ax, view[i])
                slices.append(index)
                
        return slices
    
    def make_view_ax_ind(self, view):
        return [x is None for x in view]
        
    def find_closest_ax_ind(self, ax, val):
        return np.searchsorted(ax, val)
        
        
class LikelihoodGridScanner:
    
    def __init__(self, truth, delta, n_grid, ax_names, model):
        
        self.model = model
        self.truth = truth
        self.ax_names = ax_names
        self.view = tuple(None if n>0 else 'default' for n in n_grid)
        self.make_scanning_grid(np.array(truth), np.array(delta), np.array(n_grid))
        
    def log_likelihood(self, theta, x):
        
        log_probability = self.model.log_likelihood_pdf()(theta, x)
        
        return np.sum(log_probability, axis=-1)
        
    def build_grid(self, list_of_vars):
        slices = tuple(slice(start, stop, step) for (start, stop, step) in list_of_vars)
        grid = np.mgrid[slices]
        self.grid = np.moveaxis(grid, 0, -1)
        
    def make_scanning_grid(self, truth, delta, n):
        
        if np.any(n % 2):
            #if any odd n is present
            print('Warning: If truth should be part of grid pick an even number of grid points in all dimensions')
        
        zeros = n==0
        delta[zeros] = 0
        n[zeros] = 1
        
        left = truth - delta
        right = truth + delta
        step = (right-left)/n
        
        right[zeros] = left[zeros] + 1e-3
        step[zeros] = 1000
        
        self.axes = tuple(np.arange(start, stop, step) for (start, stop, step) in zip(left, right, step))
        self.build_grid(zip(left, right, step))
        
    def scan_likelihood(self, data):

        grid_flattened = np.reshape(self.grid, (-1, self.grid.shape[-1]))
        llh_vals_flat = np.empty(grid_flattened.shape[0])

        for i,param in enumerate(grid_flattened):
            llh_vals_flat[i] = self.log_likelihood(param, data)

        llh_vals = np.reshape(llh_vals_flat, self.grid.shape[:-1])
        
        return llh_vals
        
    def __call__(self, data):
        
        llh_vals = self.scan_likelihood(data)
        
        return LikelihoodScan(self.truth, llh_vals, self.axes, 
                                self.ax_names, None).make_view(self.view)


class FitResult:
    
    def __init__(self, best_fit, errors, confidence_levels, llh_vals, llh_scan):
        self.best_fit_view = best_fit
        self.errors_view = errors
        self.best_fit = best_fit
        self.errors = errors
        self.confidence_levels = confidence_levels
        self.llh_vals = llh_vals
        self.llh_scan = llh_scan
        self.generate_best_fit()
        
    def generate_best_fit(self):
        
        missing = self.llh_scan.fixed_view_parameters
        
        if missing is not None:
        
            n_missing = len(missing)
            error_shape = self.errors_view.shape

            best_fit = np.empty(len(self.best_fit_view)+n_missing)
            errors = np.empty(shape=(error_shape[0], error_shape[1]+n_missing, error_shape[2]))

            vals_missing = [missing[k][0] for k in missing.keys()]
            inds_missing = [missing[k][1] for k in missing.keys()]

            ind = np.arange(len(self.best_fit_view))

            for k in missing.keys():
                ind[ind>=missing[k][1]] +=1

            best_fit[ind] = self.best_fit_view
            best_fit[inds_missing] = vals_missing

            errors[:,ind,:] = self.errors_view
            errors[:,inds_missing,:] = 0

            self.best_fit = best_fit
            self.errors = errors
            
        else:
            self.best_fit = self.best_fit_view
            self.errors = self.errors_view
            
    def get_resolution(self):
        
        return np.sum(self.errors, axis=-1)
            
    
class GridFitter:
    
    def __init__(self, confidence_levels):
        
        self.confidence_levels = confidence_levels
        
    def get_confidence_threshold(self, llh_max, confidence, parameters_of_interest):
        return llh_max - 0.5*chi2.ppf(confidence, df=parameters_of_interest)
        
    def get_bounding_box(self, llh, level):
    
        ind = llh>level

        bounds = []
        
        if llh.ndim==1:
            first = np.argmax(ind)
            last = ind.size - np.argmax(ind[::-1]) - 1
            bounds.append([first, last])
        
        else:
            
            new_shape = tuple(i + 1 for i in ind.shape)
            ind_new = np.zeros(new_shape)
            slices = tuple(slice(1, None, 1) for i in range(ind_new.ndim))
            ind_new[slices] = ind
            
            for i in range(llh.ndim):
                max_i = np.argmax(ind_new, axis=i-1)

                ind_i = max_i>0
                while ind_i.ndim>1:
                    max_i = np.argmax(max_i, axis=-1)
                    ind_i = max_i>0
            
                first = np.argmax(ind_i) -1
                last = max_i.shape[0] - np.argmax(ind_i[::-1]) - 2

                bounds.append([first, last])

        return np.array(bounds)
        
    def get_max_llh(self, llh):
        max_ind = np.unravel_index(llh.argmax(), llh.shape)
        max_llh = llh[max_ind]
        return max_llh, max_ind
            
    def transform_ax_ind_to_val(self, ind, axes):
    
        vals = np.empty(ind.shape)
        for i, ax in enumerate(axes):
            vals[:,i,:] = ax[ind[:,i,:]]
        
        return vals
        
    def check_bounds_warning(self, bounds, axes):
    
        for i, ax in enumerate(axes):
            if len(ax)>1:
                if np.any(bounds[:,i,0]==0) or np.any(bounds[:,i,1]==ax.size):
                    #print(f'Warning: Found bounds ({bounds[i,0]}, {bounds[i,1]}) for an axis of size {ax.size}')
                    print('Warning: Some confidence region is likely out of fitting range. Increase the fitting range!')
                    
    def check_zero_error_warning(self, bounds, max_ind):
        
        max_ind_np = np.array(max_ind)
        
        non_zero_axes = max_ind_np != 0
        
        lower_error_ind = max_ind_np[non_zero_axes]-bounds[:,non_zero_axes,0]
        upper_error_ind = bounds[:,non_zero_axes,1] - max_ind_np[non_zero_axes]
        
        if np.any(lower_error_ind==0) or np.any(upper_error_ind==0):
            print('Warning: Some confidence region yields zero error. Increase the grid resolution!')
            
                
    def make_profile_llh(self, llh_scan, parameters_of_interest):
        
        axes_tp = tuple(i for i in range(parameters_of_interest, llh_scan.llh.ndim))
        profile_llh = np.max(llh_scan.llh, axis=axes_tp)
        
        #~ profile_truth = llh_scan.truth[:parameters_of_interest]
        #~ profile_axes = llh_scan.axes[:parameters_of_interest]
        #~ profile_ax_names = llh_scan.ax_names[:parameters_of_interest]
        
        #~ profile_llh_scan = LikelihoodScan(profile_truth, profile_llh, profile_axes, 
                                #~ profile_ax_names, None)
                                
        max_llh, max_ind = self.get_max_llh(llh_scan.llh)
        
        view = tuple(None if i<parameters_of_interest else llh_scan.axes[i][ind_i] for i, ind_i in enumerate(max_ind))
        
      #  print('view', view)
                                
        profile_llh_scan = llh_scan.make_view(view)
        
        profile_llh_scan.llh = profile_llh
        
        return profile_llh_scan
            
    def get_best_fit_with_errors(self, llh_scan, parameters_of_interest=1):
        
        llh = llh_scan.llh
        axes = llh_scan.axes
        
        if not 0 < parameters_of_interest <= len(axes):
            raise ValueError('parameters_of_interest has to be at least 1 and cannot exceed the total number of parameters')
        
        profile_llh_scan = self.make_profile_llh(llh_scan, parameters_of_interest)
        
        return self.get_best_fit_with_errors_profile(profile_llh_scan)
        
    def get_best_fit_with_errors_profile(self, llh_scan):
        """
        If parameters_of_interest=n it is assumed that the first n parameters
        defined by llh_scan.axes are those of interest, the remaining parameters
        are nuisance parameters.
        """
        
        llh = llh_scan.llh
        axes = llh_scan.axes
        
        max_llh, max_ind = self.get_max_llh(llh)
        max_ind_np = np.array(max_ind)
        
        llh_vals = []
        
        bounds_ind = np.empty((len(self.confidence_levels), len(axes), 2), dtype=np.int64)
        for i, level in enumerate(self.confidence_levels):
            threshold = self.get_confidence_threshold(max_llh, level, len(axes))
            llh_vals.append(threshold)
            bounds_ind[i] = self.get_bounding_box(llh, threshold)
     
        self.check_bounds_warning(bounds_ind, axes)
        self.check_zero_error_warning(bounds_ind, max_ind)
        bound_vals = self.transform_ax_ind_to_val(bounds_ind, axes)
        
        best_fit = self.transform_ax_ind_to_val(np.expand_dims(max_ind_np,(0,-1)), axes)
        
        errors = np.empty(bound_vals.shape)
        
        errors[...,0] = best_fit[...,0] - bound_vals[...,0]
        errors[...,1] = bound_vals[...,1]-best_fit[...,0]
        
        llh_vals.append(max_llh)

        return FitResult(best_fit[0,:,0], errors, self.confidence_levels, 
                        np.array(llh_vals), llh_scan)


def add_noise(data, likelihood_model):
    
    n = data.size
    
    noise = likelihood_model.gen_noise(n)
    
    return data + noise.reshape(data.shape)
    

def get_parameter_resolution(theta, likelihood_model, n_repeat, 
                             confidence_levels, delta, n, ax_names):
    
    fitter = GridFitter(confidence_levels)
    llh_scanner = LikelihoodGridScanner(theta, delta, n, ax_names, likelihood_model)
    
    resolutions = np.empty(shape=(n_repeat, len(confidence_levels), len(theta)))
    
    model_data = likelihood_model.f(*theta)
    
    for i in range(n_repeat):
        data = add_noise(model_data, likelihood_model)
        llh_scan = llh_scanner(data)
        fit_result = fitter.get_best_fit_with_errors(llh_scan)
        resolutions[i] = fit_result.get_resolution()
    
    res = np.mean(resolutions, axis=0)
    res_err = np.sqrt(np.var(resolutions, axis=0)/n_repeat)
    
    return res, res_err
    
    
