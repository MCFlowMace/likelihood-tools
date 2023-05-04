
"""

Author: F. Thomas
Date: September 22, 2022

"""

import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
from tqdm import tqdm
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import root_scalar, minimize

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
    
    def __init__(self, f, sigma, complex_data=False):
        LikelihoodModel.__init__(self, f)
        self.sigma = sigma
        self.complex_data = complex_data
        
    #def likelihood_pdf(self):
    #    return lambda theta,x: norm.pdf(x,scale=self.sigma, loc=self.f(*theta))
    def log_likelihood_pdf(self):
        sigma = self.sigma
        if self.complex_data:
            sigma = sigma/2
        return lambda theta,x: -0.5*np.abs((x - self.f(*theta))/sigma)**2
        
    def gen_noise(self, n):
        
        if self.complex_data:
             return np.random.normal(loc=0, scale=self.sigma/2, size=(n, 2)).view(np.complex128)
        else:
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
    
    def __init__(self, truth, llh, axes, ax_names, fixed_view_parameters,
                    llh_f=None):
        
        self.truth = truth
        self.llh = llh
        self.axes = axes
        self.ax_names = ax_names
        self.fixed_view_parameters = fixed_view_parameters
        self.llh_f = llh_f
        self.dim = len(axes)
        
    def get_original_param_index(self):
        ind = np.arange(self.dim)
        missing = self.fixed_view_parameters
        
        if missing is not None:
            for k in missing.keys():
                ind[ind>=missing[k][1]] +=1
            
        return ind
        
    def make_view(self, view=None):
        
        if view is not None:
            view_slices = self.make_view_slices(view)
            view_ax_ind = self.make_view_ax_ind(view)

            truth = [t for i, t in enumerate(self.truth) if view_ax_ind[i]]
            llh = self.llh
            if llh is not None:
                llh = llh[tuple(view_slices)]
            axes = tuple(ax for i, ax in enumerate(self.axes) if view_ax_ind[i])
            ax_names = [n for i, n in enumerate(self.ax_names) if view_ax_ind[i]]
            
            orig_ind= self.get_original_param_index()
            
            fixed_view_parameters = {name: (self.axes[i][view_slices[i]], orig_ind[i]) for i, name in enumerate(self.ax_names) if not view_ax_ind[i]}
            
            if self.fixed_view_parameters is not None:
                fixed_view_parameters_old = self.fixed_view_parameters.copy()
                fixed_view_parameters_old.update(fixed_view_parameters)
                fixed_view_parameters = fixed_view_parameters_old


            def f_bar(x):
                c = 0
                param = []
                for i, scan_ind in enumerate(view_ax_ind):
                    if not scan_ind:
                        param.append(self.truth[i])
                    else:
                        param.append(x[c])
                        c+=1
                return np.squeeze(self.llh_f(param))
                
            if self.llh_f is not None:
                f_view = f_bar
            else:
                f_view = None
                
            return LikelihoodScan(truth, llh, axes, ax_names, 
                                    fixed_view_parameters, f_view)
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
        
    def interpolate(self, interpolation_axes=None):

        if interpolation_axes is None:
            axes = self.axes
        else:
            axes = interpolation_axes
        
        if len(axes)>2:
            raise ValueError('Can only interpolate 1D and 2D scans')
            
        if len(axes)==2:
            interpolation = interp2d(axes[0], axes[1], self.llh.transpose(), kind='cubic', bounds_error=False)
            self.llh_f = lambda x: interpolation(x[0], x[1]).transpose()
        else:
            interpolation = interp1d(axes[0], self.llh, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.llh_f = lambda x: interpolation(x[0])
        

class LikelihoodScanner(ABC):
    
    def __init__(self, truth, delta, n_eval, ax_names, model):
        
        self.model = model
        self.truth = truth
        self.ax_names = ax_names

        self.delta = delta
        self.view_ = tuple(None if n>0 else 'default' for n in n_eval)
        self.axes = LikelihoodScanner.make_axes(truth, delta, n_eval)
        self.dim_ = sum([n>0 for n in n_eval])

    def view(self):
        return self.view_

    def dim(self):
        return self.dim_
        
    def get_axes(list_of_vars):
        return tuple(np.arange(start, stop, step) for (start, stop, step) in list_of_vars)
    
    def convert_arrays(truth, delta, n):
        delta_np = np.array([[a,a] if type(a) is not list else a for a in delta])
        return np.array(truth), delta_np, np.array(n)
    
    def get_left_right_step(truth_a, delta_a, n_a):

        truth, delta, n = LikelihoodScanner.convert_arrays(truth_a, delta_a, n_a)
        #if np.any(n % 2):
        if np.any((n!=0)&(n%2==0)):
            #if any even n is present
            print('Warning: If truth should be part of grid pick an odd number of grid points in all dimensions')
        
        zeros = n==0
        delta[zeros] = 0
        n[zeros] = 1
        
        left = truth - delta[...,0]
        right = truth + delta[...,1]
        step = np.ones_like(truth)
        step[~zeros] = (delta[~zeros,0]+delta[~zeros,1])/(n[~zeros]-1)
        
        right = right + step

        return left, right, step
        
    def make_axes(truth, delta, n):
        left, right, step = LikelihoodScanner.get_left_right_step(truth, delta, n)
        return LikelihoodScanner.get_axes(zip(left, right, step))

    @abstractmethod
    def scan_likelihood(self, data):
        pass
        
    def log_likelihood(self, theta, x):
        
        log_probability = self.model.log_likelihood_pdf()(theta, x)
        
        return np.sum(log_probability, axis=-1)
    
    def __call__(self, data):
    
        return self.scan_likelihood(data)
    
    def get_param_view(self, param):
        
        view_ax_ind = [x is None for x in self.view()]
        return [t for i, t in enumerate(param) if view_ax_ind[i]]
    
    def get_asimov_scan(self):
        data = self.model.f(*self.truth)
        return self(data)
    

class LikelihoodGridScanner(LikelihoodScanner):
    
    def __init__(self, truth, delta, n_eval, ax_names, model, n_grid):
        LikelihoodScanner.__init__(self, truth, delta, n_eval, ax_names, model)
        self.scanning_grid = LikelihoodGridScanner.make_scanning_grid(truth, delta, n_grid)
        self.scanning_axes = LikelihoodScanner.make_axes(truth, delta, n_grid)
        #self.interpolate = n_grid!=n_eval

    def get_grid(list_of_vars):
        slices = tuple(slice(start, stop, step) for (start, stop, step) in list_of_vars)
        grid = np.mgrid[slices]
        return np.moveaxis(grid, 0, -1)
    
    def make_scanning_grid(truth, delta, n):
        left, right, step = LikelihoodScanner.get_left_right_step(truth, delta, n)
        return LikelihoodGridScanner.get_grid(zip(left, right, step))
        
    def scan_likelihood(self, data):
        
        llh_vals = self.scan_likelihood_grid(data, self.scanning_grid)
        
        llh_scan = LikelihoodScan(self.truth, llh_vals, self.axes, 
                                    self.ax_names, None, None).make_view(self.view())
        
        #if self.interpolate:
        axes = [self.scanning_axes[i] for i,x in enumerate(self.view()) if x is None]
        llh_scan.interpolate(interpolation_axes=axes)
        llh_scan.llh = llh_scan.llh_f(llh_scan.axes)

        return llh_scan
        
    def scan_likelihood_grid(self, data, grid):
        
        grid_flattened = np.reshape(grid, (-1, grid.shape[-1]))
        llh_vals_flat = np.empty(grid_flattened.shape[0])

        for i, param in enumerate(pbar:= tqdm(grid_flattened)):
            pbar.set_postfix_str(str(self.get_param_view(param)))
            llh_vals_flat[i] = self.log_likelihood(param, data)

        llh_vals = np.reshape(llh_vals_flat, grid.shape[:-1])
        
        return llh_vals

"""                    
    def get_asimov_scan(self, approximation_samples=None):
        
        if approximation_samples is None:
            data = self.model.f(*self.truth)
        
            return self(data)
            
        elif approximation_samples==1:
            
            llh_f = self.approximate_llh_2order()
            
            llh_scan = LikelihoodScan(self.truth, None, self.axes, 
                                    self.ax_names, None, None).make_view(self.view)
                                    
            llh_scan.llh_f = lambda x: llh_f(x[0])
            
            return llh_scan
            
        elif approximation_samples==3:
            
            llh_f = self.approximate_llh_4order()
            
            llh_scan = LikelihoodScan(self.truth, None, self.axes, 
                                    self.ax_names, None, None).make_view(self.view)
                                    
            llh_scan.llh_f = lambda x: llh_f(x[0])
            
            return llh_scan
            
        elif approximation_samples>3 and not approximation_samples%2==0:
            
            n = [approximation_samples if x is None else 0 for x in self.view]
            
            grid, axes = self.make_scanning_grid(np.array(self.truth), np.array(self.delta), np.array(n))
            data = self.model.f(*self.truth)
            
            return self.scan_likelihood(data, grid=grid, axes=axes, interpolate=True, keep_scan=False)
        else:
            raise ValueError('Invalid number of approximation samples')
        
    def get_offcenter_param(self, relative_offset):
        relative_offset = relative_offset+0.5
        ind = tuple(0 if x==1 else int(x*relative_offset) for x in self.grid.shape[:-1])
        
        return self.grid[ind]
        
    def approximate_llh_2order(self, relative_offset=0.05):
        
        truth_view = self.get_param_view(self.truth)
        
        if self.dim>1:
            raise ValueError('Can only approximate 1D llh scans')
        
        y_asimov = self.model.f(*self.truth)
        
        theta_a = truth_view[0]
        #theta_bar = theta_a*(1 + delta)

        param_bar = self.get_offcenter_param(relative_offset)
        
        theta_bar = self.get_param_view(param_bar)[0]
        
        alpha = self.log_likelihood(param_bar, y_asimov)
        
        a = alpha/(theta_bar - theta_a)**2
        
        return lambda theta: a*(theta-theta_a)**2
        
    def approximate_llh_4order(self, relative_offset=0.05):
        
        truth_view = self.get_param_view(self.truth)
        
        if self.dim>1:
            raise ValueError('Can only approximate 1D llh scans')
           
        y_asimov = self.model.f(*self.truth)
        
        theta_a = truth_view[0]
        
        param_1 = self.get_offcenter_param(relative_offset)
        param_2 = self.get_offcenter_param(-relative_offset)
        param_3 = self.get_offcenter_param(2*relative_offset)
        
        theta_1 = self.get_param_view(param_1)[0]
        theta_2 = self.get_param_view(param_2)[0]
        theta_3 = self.get_param_view(param_3)[0]
        
        alpha = self.log_likelihood(param_1, y_asimov)
        beta = self.log_likelihood(param_2, y_asimov)
        gamma = self.log_likelihood(param_3, y_asimov)
        
        d12 = theta_1-theta_2
        d13 = theta_1-theta_3
        d23 = theta_2-theta_3
        da1 = theta_a-theta_1
        da2 = theta_a-theta_2
        da3 = theta_a-theta_3
        
        d = d12*d23
        
        a = (-beta/da2**2 + ((gamma*d12)/da3**2 + alpha*d23/da1**2)/d13)/d
        
        b = (beta*(2*theta_a+theta_1+theta_3)/da2**2
             +(-(gamma*d12*(2*theta_a+theta_1+theta_2))/da3**2
               +(alpha*(theta_3-theta_2)*(2*theta_a+theta_2+theta_3))/da1**2)/d13)/d
        
        c = (-beta*(theta_a**2+theta_1*theta_3+2*theta_a*(theta_1+theta_3))/da2**2
            +((gamma*d12*(theta_a**2 + theta_1*theta_2+2*theta_a*(theta_1+theta_2)))/da3**2
             +(alpha*d23*(theta_a**2+theta_2*theta_3+2*theta_a*(theta_2+theta_3)))/da1**2)/d13)/d
        
        f = lambda theta: (c*(theta-theta_a)**2 + a*(theta**4 + 3*theta_a**4 - 4*theta_a**3*theta)
                             +b*(theta**3 + 2*theta_a**3 - 3*theta*theta_a**2))
        
        return f
"""

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

            #ind = np.arange(len(self.best_fit_view))

            #for k in missing.keys():
            #    ind[ind>=missing[k][1]] +=1
            
            ind = self.llh_scan.get_original_param_index()

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
    

def sigma_to_confidence(n_sigma):
    return norm.cdf(n_sigma)-norm.cdf(-n_sigma)


def confidence_to_threshold(llh_max, confidence, parameters_of_interest):
    return llh_max - 0.5*chi2.ppf(confidence, df=parameters_of_interest)

            
class Fitter(ABC):
    
    def __init__(self, sigma_levels):
        
        self.confidence_levels = [sigma_to_confidence(sigma) for sigma in sigma_levels]
        
   # def get_confidence_threshold(self, llh_max, confidence, parameters_of_interest):
   #     return llh_max - 0.5*chi2.ppf(confidence, df=parameters_of_interest)
        
    def get_best_fit_with_errors(self, llh_scan, parameters_of_interest=1):
        
        axes = llh_scan.axes
        
        if not 0 < parameters_of_interest <= len(axes):
            raise ValueError('parameters_of_interest has to be at least 1 and cannot exceed the total number of parameters')
        
        profile_llh_scan = self.make_profile_llh(llh_scan, parameters_of_interest)
        
        return self.get_best_fit_with_errors_profile(profile_llh_scan)
        
    @abstractmethod
    def make_profile_llh(self, llh_scan, parameters_of_interest):
        pass
    
    @abstractmethod    
    def get_best_fit_with_errors_profile(self, llh_scan):
        pass
             
             
class FunctionFitter(Fitter):
    
    def __init__(self, sigma_levels, asimov=True):
        Fitter.__init__(self, sigma_levels)
        
        if not asimov:
            raise NotImplementedError('This fitter for non-asimov datasets is not implemented')
        
    def make_profile_llh(self, llh_scan, parameters_of_interest):
        
        if llh_scan.llh_f is None:
            raise ValueError('Needs a likelihood function')
            
        if parameters_of_interest>1:
            raise NotImplementedError('No implementation for profile likelihood with more than 1 parameter of interest')
            
        if llh_scan.dim==2:
            profile_llh = np.empty(llh_scan.axes[1].shape)
            
            dx = llh_scan.axes[0][1]-llh_scan.axes[0][0]
            x_pos = np.empty(llh_scan.axes[1].shape)
            for i, y in enumerate(llh_scan.axes[1]):
                minimization = minimize(lambda x: -llh_scan.llh_f([x, y]), x0=llh_scan.truth[0], 
                               method='Powell',
                                bounds=[[llh_scan.axes[0][0],llh_scan.axes[0][-1]]])
                x = minimization.x[0]
                profile_llh[i] = llh_scan.llh_f([x, y])
                x_pos[i] = x
                
            ind = np.argsort(x_pos)
            x_pos_sort = x_pos[ind]
            profile_llh_sort = profile_llh[ind]

            first = np.argmax(x_pos_sort-llh_scan.axes[0][0]>dx)
            last = len(x_pos) - np.argmax(dx<llh_scan.axes[0][-1]-x_pos_sort[::-1])

            f = interp1d(x_pos_sort[first:last], profile_llh_sort[first:last], kind='cubic', bounds_error=False, fill_value='extrapolate')

            view = tuple(None if i<parameters_of_interest else llh_scan.truth[i] for i in range(llh_scan.dim))
                                    
            profile_llh_scan = llh_scan.make_view(view)
            
            profile_llh_scan.llh_f = lambda x: f(x[0])
            profile_llh_scan.llh = profile_llh_scan.llh_f(profile_llh_scan.axes)
            
        else:
            profile_llh_scan = llh_scan
        
        return profile_llh_scan
        
    def find_roots(self, llh_scan, threshold):
        
        left = root_scalar(lambda x: llh_scan.llh_f([x])-threshold, 
                            method='secant', x0=llh_scan.axes[0][0], x1=llh_scan.truth[0]).root
                            
        right = root_scalar(lambda x: llh_scan.llh_f([x])-threshold, 
                            method='secant', x0=llh_scan.truth[0], x1=llh_scan.axes[0][-1]).root
        
        return np.array([left, right])
        
    def get_best_fit_with_errors_profile(self, llh_scan):
        """
        If parameters_of_interest=n it is assumed that the first n parameters
        defined by llh_scan.axes are those of interest, the remaining parameters
        are nuisance parameters.
        """
        
        if llh_scan.dim != 1:
            raise ValueError('Root finding based fitter needs 1D llh scan')
        
        axes = llh_scan.axes
        
        max_llh = 0
        
        llh_vals = []
        
        bound_vals = np.empty((len(self.confidence_levels), len(axes), 2))
        for i, level in enumerate(self.confidence_levels):
            threshold = confidence_to_threshold(max_llh, level, len(axes))
            llh_vals.append(threshold)
            bound_vals[i] = self.find_roots(llh_scan, threshold)
        
        best_fit = np.expand_dims(llh_scan.truth,(0,-1))
        
        errors = np.empty(bound_vals.shape)
        
        errors[...,0] = best_fit[...,0] - bound_vals[...,0]
        errors[...,1] = bound_vals[...,1]-best_fit[...,0]
        
        llh_vals.append(max_llh)

        return FitResult(best_fit[0,:,0], errors, self.confidence_levels, 
                        np.array(llh_vals), llh_scan)
                        
                        
class GridFitter(Fitter):
    
    def __init__(self, sigma_levels):
        Fitter.__init__(self, sigma_levels)
        
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
        
        if llh_scan.llh is None:
            if llh_scan.llh_f is None:
                raise ValueError('Needs a scanned likelihood on a grid or a likelihood function')
            llh_scan.llh = llh_scan.llh_f(llh_scan.axes)
        
        axes_tp = tuple(i for i in range(parameters_of_interest, llh_scan.llh.ndim))
        profile_llh = np.max(llh_scan.llh, axis=axes_tp)
        
        #~ profile_truth = llh_scan.truth[:parameters_of_interest]
        #~ profile_axes = llh_scan.axes[:parameters_of_interest]
        #~ profile_ax_names = llh_scan.ax_names[:parameters_of_interest]
        
        #~ profile_llh_scan = LikelihoodScan(profile_truth, profile_llh, profile_axes, 
                                #~ profile_ax_names, None)
                                
        max_llh, max_ind = self.get_max_llh(llh_scan.llh)
        
        view = tuple(None if i<parameters_of_interest else llh_scan.axes[i][ind_i] for i, ind_i in enumerate(max_ind))
          
        profile_llh_scan = llh_scan.make_view(view)
        
        profile_llh_scan.llh = profile_llh

        if profile_llh_scan.llh_f is not None:
            profile_llh_scan.interpolate()
        
        return profile_llh_scan
        
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
            threshold = confidence_to_threshold(max_llh, level, len(axes))
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
    
